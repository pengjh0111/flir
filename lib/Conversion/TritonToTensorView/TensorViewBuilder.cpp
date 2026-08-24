//===- TensorViewBuilder.cpp - Direct TensorView construction ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToTensorView/TensorViewBuilder.h"

#include "triton-shared/Conversion/TritonToTensorView/Matcher.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <string>

using namespace mlir;
namespace tv = mlir::triton::tv;

namespace {

static Type getPtrPointeeType(Type type) {
  if (auto ptr = dyn_cast<tv::PtrType>(type))
    return ptr.getPointeeType();
  if (auto ptr = dyn_cast<triton::PointerType>(type))
    return ptr.getPointeeType();
  return Type();
}

static Value buildBaseView(OpBuilder &b, Location loc, Value basePtr,
                           Type elementType, ArrayRef<int64_t> strideStatic,
                           ArrayRef<Value> strideDyn, ArrayRef<Value> extent) {
  basePtr = tv::ensureTvPtr(basePtr);
  unsigned rank = strideStatic.size();

  SmallVector<Value> strideOperands;
  SmallVector<int64_t> strideTy;
  for (unsigned d = 0; d < rank; ++d) {
    if (strideStatic[d] == ShapedType::kDynamic) {
      strideOperands.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), strideDyn[d]));
      strideTy.push_back(ShapedType::kDynamic);
    } else {
      strideOperands.push_back(
          b.create<arith::ConstantIndexOp>(loc, strideStatic[d]));
      strideTy.push_back(strideStatic[d]);
    }
  }

  SmallVector<Value> sizeOperands;
  for (unsigned d = 0; d < rank; ++d) {
    if (d < extent.size() && extent[d])
      sizeOperands.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), extent[d]));
    else
      sizeOperands.push_back(b.create<arith::ConstantIndexOp>(
          loc, std::numeric_limits<int64_t>::max()));
  }

  SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
  auto baseTy = tv::TensorViewType::get(dynShape, elementType, strideTy,
                                        /*encoding=*/Attribute());
  return b
      .create<tv::MakeTensorViewOp>(loc, baseTy, basePtr, sizeOperands,
                                    strideOperands)
      .getResult();
}

static Value attachViewEncoding(OpBuilder &b, Location loc, Value baseView,
                                ArrayRef<int64_t> tile,
                                ArrayRef<int64_t> traversal,
                                ArrayRef<int64_t> sparseDims) {
  MLIRContext *ctx = b.getContext();
  auto baseTy = cast<tv::TensorViewType>(baseView.getType());
  ArrayRef<int64_t> strideTy = baseTy.getStrides();
  Type elementType = baseTy.getElementType();
  SmallVector<int64_t> dynShape(tile.size(), ShapedType::kDynamic);

  if (!sparseDims.empty()) {
    Attribute enc = tv::GatherScatterViewAttr::get(ctx, tile, sparseDims);
    auto viewTy = tv::TensorViewType::get(dynShape, elementType, strideTy, enc);
    return b.create<tv::MakeGatherScatterViewOp>(loc, viewTy, baseView)
        .getResult();
  }

  bool isPartition = traversal == tile;
  Attribute enc =
      isPartition ? Attribute(tv::PartitionViewAttr::get(ctx, tile))
                  : Attribute(tv::StridedViewAttr::get(ctx, tile, traversal));
  auto viewTy = tv::TensorViewType::get(dynShape, elementType, strideTy, enc);
  if (isPartition)
    return b.create<tv::MakePartitionViewOp>(loc, viewTy, baseView).getResult();
  return b.create<tv::MakeStridedViewOp>(loc, viewTy, baseView).getResult();
}

static SmallVector<Value> castIndices(OpBuilder &b, Location loc,
                                      ValueRange rawIndices,
                                      ArrayRef<int64_t> sparseDims) {
  SmallVector<Value> indices;
  for (auto [d, idx] : llvm::enumerate(rawIndices)) {
    if (llvm::is_contained(sparseDims, d)) {
      indices.push_back(idx);
      continue;
    }
    indices.push_back(
        idx.getType().isIndex()
            ? idx
            : b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx));
  }
  return indices;
}

static bool isAddptrChainToFuncArg(Value value, BlockArgument &argOut) {
  Value current = value;
  while (auto addptr = current.getDefiningOp<triton::AddPtrOp>()) {
    if (isa<RankedTensorType>(addptr.getOffset().getType()))
      return false;
    current = addptr.getPtr();
  }
  if (current == value)
    return false;
  auto arg = dyn_cast<BlockArgument>(current);
  if (!arg || !isa<triton::FuncOp>(arg.getOwner()->getParentOp()))
    return false;
  argOut = arg;
  return true;
}

} // namespace

namespace mlir {
namespace triton {
namespace tv {

Value createTensorViewBase(OpBuilder &b, Location loc, Value basePtr,
                           ValueRange shape, ValueRange strides) {
  BlockArgument offendingArg;
  if (isAddptrChainToFuncArg(basePtr, offendingArg)) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "tl.tensor_view: `base` is computed via raw pointer arithmetic "
          "on function argument #"
       << offendingArg.getArgNumber()
       << " (e.g. `arg + batch_idx*stride`) -- this is not supported. "
          "Fold the offset into an extra leading dimension of "
          "`shape`/`strides` instead (covering the whole logical tensor, "
          "e.g. adding a batch/head dimension) and pass that position via "
          "`index=` to `tl.load`/`tl.store`.";
    os.flush();
    mlir::emitError(loc) << msg;
    llvm::report_fatal_error(llvm::StringRef(msg));
  }

  unsigned rank = shape.size();
  Type elementType = getPtrPointeeType(basePtr.getType());
  SmallVector<int64_t> strideStatic(rank, ShapedType::kDynamic);
  SmallVector<Value> strideDyn(strides.begin(), strides.end());
  SmallVector<Value> extent(shape.begin(), shape.end());
  return buildBaseView(b, loc, basePtr, elementType, strideStatic, strideDyn,
                       extent);
}

Value tensorViewLoad(OpBuilder &b, Location loc, Value baseView,
                     ArrayRef<int64_t> tile, ArrayRef<int64_t> traversal,
                     ArrayRef<int64_t> sparseDims, ValueRange index,
                     Type resultTy, Value mask) {
  Value view = attachViewEncoding(
      b, loc, baseView, tile, traversal.empty() ? tile : traversal, sparseDims);
  return b
      .create<tv::ViewLoadOp>(loc, resultTy, view,
                              castIndices(b, loc, index, sparseDims), mask)
      .getResult();
}

void tensorViewStore(OpBuilder &b, Location loc, Value baseView,
                     ArrayRef<int64_t> tile, ArrayRef<int64_t> traversal,
                     ArrayRef<int64_t> sparseDims, ValueRange index,
                     Value value, Value mask) {
  Value view = attachViewEncoding(
      b, loc, baseView, tile, traversal.empty() ? tile : traversal, sparseDims);
  b.create<tv::ViewStoreOp>(loc, view, value,
                            castIndices(b, loc, index, sparseDims), mask);
}

} // namespace tv
} // namespace triton
} // namespace mlir
