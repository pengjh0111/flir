//===- TensorViewLowering.cpp - tv access ops -> generic memref
//-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower TensorView operations to portable memref, bufferization, scf, and
// tensor operations. Address spaces and DMA selection remain backend concerns.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TensorViewLowering/Passes.h"

#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TENSORVIEWLOWERING
#include "triton-shared/Conversion/TensorViewLowering/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace tv = mlir::triton::tv;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Information recovered from an encoded view chain.
struct ViewData {
  Value base; // flat memref<?xT> (plain, no address space)
  Type elementType;
  unsigned rank = 0;
  SmallVector<int64_t> tile; // per-dim tile size
  SmallVector<int64_t>
      traversal; // per-dim traversal stride (== tile for partition)
  SmallVector<int64_t> strideStatic; // per-dim element stride, or kDynamic
  SmallVector<Value> strideVal;      // per-dim element stride SSA (index)
  SmallVector<Value> sizeVal;        // per-dim logical extent SSA (index)
  SmallVector<int64_t> sparseDims;   // gather/scatter dimensions (one or more)
  tv::PaddingValue paddingValue = tv::PaddingValue::ZERO;
  bool needsBaseCast = false;
  bool ok = false;

  bool isGatherScatter() const { return !sparseDims.empty(); }
};

static ViewData parseView(Value viewVal) {
  ViewData vi;
  auto encTy = dyn_cast<tv::TensorViewType>(viewVal.getType());
  if (!encTy)
    return vi;
  Attribute encoding = encTy.getEncoding();
  if (!tv::isViewEncoding(encoding))
    return vi;
  SmallVector<int64_t> tile = tv::getEncodingTileShape(encoding);
  SmallVector<int64_t> traversal = tv::getEncodingTraversal(encoding);
  if (tile.size() != traversal.size())
    return vi;
  vi.sparseDims = tv::getEncodingSparseDims(encoding);
  vi.paddingValue = tv::getEncodingPaddingValue(encoding);
  if (isa<tv::GatherScatterViewAttr>(encoding) && vi.sparseDims.empty())
    return vi;

  // Follow the encoded view to its base view.
  auto viewOp = dyn_cast_or_null<tv::TensorViewEncodingOpInterface>(
      viewVal.getDefiningOp());
  if (!viewOp)
    return vi;
  Value baseView = viewOp.getSource();

  auto mtv = baseView.getDefiningOp<tv::MakeTensorViewOp>();
  if (!mtv)
    return vi;
  auto baseTy = dyn_cast<tv::TensorViewType>(mtv.getResult().getType());
  if (!baseTy)
    return vi;
  // Read the source untyped after function arguments have become memrefs.
  Value base = mtv->getOperand(0);
  // Record bridge casts introduced for non-block-argument pointer bases.
  // Materialization is deferred until after a pattern has fully matched.
  auto bridgeCast = base.getDefiningOp<UnrealizedConversionCastOp>();
  if (bridgeCast && bridgeCast.getNumOperands() == 1 &&
      isa<triton::PointerType>(bridgeCast.getOperand(0).getType())) {
    Value source = bridgeCast.getOperand(0);
    if (source.getDefiningOp<triton::AddPtrOp>())
      return vi;
    base = source;
    vi.needsBaseCast = true;
  }
  if (!vi.needsBaseCast && !isa<MemRefType>(base.getType()))
    return vi;

  unsigned rank = tile.size();
  if (baseTy.getStrides().size() != rank || mtv.getSizes().size() != rank ||
      mtv.getStrides().size() != rank)
    return vi;

  vi.base = base;
  vi.elementType = baseTy.getElementType();
  vi.rank = rank;
  vi.tile.assign(tile.begin(), tile.end());
  vi.traversal.assign(traversal.begin(), traversal.end());
  vi.strideStatic.assign(baseTy.getStrides().begin(),
                         baseTy.getStrides().end());
  vi.strideVal.assign(mtv.getStrides().begin(), mtv.getStrides().end());
  vi.sizeVal.assign(mtv.getSizes().begin(), mtv.getSizes().end());
  vi.ok = true;
  return vi;
}

static Value materializeBase(OpBuilder &builder, Location loc,
                             const ViewData &view) {
  if (!view.needsBaseCast)
    return view.base;
  auto type = MemRefType::get({ShapedType::kDynamic}, view.elementType);
  return builder.create<UnrealizedConversionCastOp>(loc, type, view.base)
      .getResult(0);
}

/// Physical layout of one GM-to-local transfer. The logical view keeps every
/// dimension, while unit tile dimensions contribute only to the scalar origin
/// offset and do not need to increase the DMA rank.
struct PhysicalTransferLayout {
  SmallVector<unsigned> logicalDims;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strideStatic;
  SmallVector<ReassociationIndices> reassociation;

  bool collapses(unsigned logicalRank) const {
    return logicalDims.size() != logicalRank;
  }
};

static std::optional<PhysicalTransferLayout>
buildPhysicalTransferLayout(const ViewData &view) {
  PhysicalTransferLayout layout;
  for (unsigned d = 0; d < view.rank; ++d) {
    if (view.tile[d] == 1)
      continue;
    layout.logicalDims.push_back(d);
    layout.shape.push_back(view.tile[d]);
    layout.strideStatic.push_back(view.strideStatic[d]);
  }

  // Keep transfers at least rank one even when the logical tile is all-unit.
  if (layout.logicalDims.empty() && view.rank != 0) {
    unsigned d = view.rank - 1;
    layout.logicalDims.push_back(d);
    layout.shape.push_back(view.tile[d]);
    layout.strideStatic.push_back(view.strideStatic[d]);
  }

  if (!layout.collapses(view.rank))
    return layout;

  auto reassociation =
      getReassociationIndicesForCollapse(view.tile, layout.shape);
  if (!reassociation)
    return std::nullopt;
  layout.reassociation = std::move(*reassociation);
  return layout;
}

/// Reinterpret the base as a GM tile. `transferLayout` selects a physical DMA
/// rank distinct from the logical view rank; a null layout preserves the full
/// logical tile shape.
static Value createGmTileView(OpBuilder &b, Location loc, const ViewData &vi,
                              Value off,
                              const PhysicalTransferLayout *transferLayout) {
  MLIRContext *ctx = b.getContext();
  if (!transferLayout) {
    SmallVector<OpFoldResult> sizes, strides;
    for (unsigned d = 0; d < vi.rank; ++d) {
      sizes.push_back(b.getIndexAttr(vi.tile[d]));
      if (vi.strideStatic[d] == ShapedType::kDynamic)
        strides.push_back(vi.strideVal[d]);
      else
        strides.push_back(b.getIndexAttr(vi.strideStatic[d]));
    }
    auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                         vi.strideStatic);
    auto gmTileTy = MemRefType::get(vi.tile, vi.elementType, layout);
    return b.create<memref::ReinterpretCastOp>(
        loc, gmTileTy, vi.base, OpFoldResult(off), sizes, strides);
  }
  SmallVector<OpFoldResult> sizes, strides;
  for (unsigned d : transferLayout->logicalDims) {
    sizes.push_back(b.getIndexAttr(vi.tile[d]));
    if (vi.strideStatic[d] == ShapedType::kDynamic)
      strides.push_back(vi.strideVal[d]);
    else
      strides.push_back(b.getIndexAttr(vi.strideStatic[d]));
  }
  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                       transferLayout->strideStatic);
  auto gmTileTy =
      MemRefType::get(transferLayout->shape, vi.elementType, layout);
  return b.create<memref::ReinterpretCastOp>(loc, gmTileTy, vi.base,
                                             OpFoldResult(off), sizes, strides);
}

/// Compute the physical tile-origin offset.
static Value buildTileOriginOffset(OpBuilder &b, Location loc,
                                   const ViewData &vi, ValueRange indices) {
  Value off;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal[d]);
    Value logical = b.create<arith::MulIOp>(loc, indices[d], cTrav);
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    off = d == 0 ? phys : b.create<arith::AddIOp>(loc, off, phys);
  }
  return off;
}

static Value asIndex(OpBuilder &b, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
}

static Value createPaddingConstant(OpBuilder &b, Location loc,
                                   const ViewData &vi) {
  if (vi.paddingValue == tv::PaddingValue::ZERO)
    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(vi.elementType));

  auto floatType = cast<FloatType>(vi.elementType);
  const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
  llvm::APFloat value =
      vi.paddingValue == tv::PaddingValue::NAN_VALUE
          ? llvm::APFloat::getNaN(semantics)
          : llvm::APFloat::getInf(semantics,
                                  vi.paddingValue == tv::PaddingValue::NEG_INF);
  return b.create<arith::ConstantOp>(loc, FloatAttr::get(floatType, value));
}

struct ElementAccess {
  Value physicalOffset;
  Value inBounds;
};

static ElementAccess buildElementAccess(OpBuilder &b, Location loc,
                                        const ViewData &vi, ValueRange indices,
                                        ArrayRef<int64_t> sparseDims,
                                        ValueRange coordinates) {
  ElementAccess access;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  access.inBounds = b.create<arith::ConstantIntOp>(loc, 1, 1);
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value logicalCoordinate;
    if (llvm::is_contained(sparseDims, static_cast<int64_t>(d))) {
      auto extracted =
          b.create<tensor::ExtractOp>(loc, indices[d], coordinates[d]);
      logicalCoordinate = asIndex(b, loc, extracted.getResult());
    } else {
      Value traversal = b.create<arith::ConstantIndexOp>(loc, vi.traversal[d]);
      Value origin = b.create<arith::MulIOp>(loc, indices[d], traversal);
      logicalCoordinate = b.create<arith::AddIOp>(loc, origin, coordinates[d]);
    }
    Value nonNegative = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                logicalCoordinate, zero);
    Value belowExtent = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, logicalCoordinate, vi.sizeVal[d]);
    Value dimensionInBounds =
        b.create<arith::AndIOp>(loc, nonNegative, belowExtent);
    access.inBounds =
        b.create<arith::AndIOp>(loc, access.inBounds, dimensionInBounds);

    Value physical =
        b.create<arith::MulIOp>(loc, logicalCoordinate, vi.strideVal[d]);
    access.physicalOffset =
        access.physicalOffset
            ? b.create<arith::AddIOp>(loc, access.physicalOffset, physical)
            : physical;
  }
  return access;
}

static Value buildAccessInBoundsCondition(OpBuilder &b, Location loc,
                                          const ViewData &vi,
                                          ValueRange indices) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value condition = b.create<arith::ConstantIntOp>(loc, 1, 1);
  for (unsigned d = 0; d < vi.rank; ++d) {
    if (llvm::is_contained(vi.sparseDims, static_cast<int64_t>(d))) {
      Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
      auto loop =
          b.create<scf::ForOp>(loc, zero, upper, one, ValueRange{condition});
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        Value coordinate =
            asIndex(b, loc,
                    b.create<tensor::ExtractOp>(
                        loc, indices[d], ValueRange{loop.getInductionVar()}));
        Value nonNegative = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, coordinate, zero);
        Value belowExtent = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, coordinate, vi.sizeVal[d]);
        Value valid = b.create<arith::AndIOp>(loc, nonNegative, belowExtent);
        b.create<scf::YieldOp>(
            loc, b.create<arith::AndIOp>(loc, loop.getRegionIterArg(0), valid)
                     .getResult());
      }
      condition = loop.getResult(0);
      continue;
    }

    Value traversal = b.create<arith::ConstantIndexOp>(loc, vi.traversal[d]);
    Value origin = b.create<arith::MulIOp>(loc, indices[d], traversal);
    Value nonNegative =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, origin, zero);
    Value tile = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    Value end = b.create<arith::AddIOp>(loc, origin, tile);
    Value withinExtent = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                                 end, vi.sizeVal[d]);
    Value dimensionInBounds =
        b.create<arith::AndIOp>(loc, nonNegative, withinExtent);
    condition = b.create<arith::AndIOp>(loc, condition, dimensionInBounds);
  }
  return condition;
}

/// Reinterpret a scalar GM element at the given offset.
static Value createDiscreteGmElementView(OpBuilder &b, Location loc, Value base,
                                         Value offset, Type elementType) {
  auto layout = StridedLayoutAttr::get(b.getContext(),
                                       /*offset=*/ShapedType::kDynamic, {1});
  auto ty = MemRefType::get({1}, elementType, layout);
  return b.create<memref::ReinterpretCastOp>(
      loc, ty, base, OpFoldResult(offset),
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

/// Compute the contiguous block geometry for one sparse index.
static void
buildContiguousBlockGeometry(OpBuilder &b, Location loc, const ViewData &vi,
                             ValueRange indices, unsigned sparseDim,
                             Value sparseIndex, Value &offset,
                             SmallVectorImpl<OpFoldResult> &sizes,
                             SmallVectorImpl<OpFoldResult> &strides,
                             SmallVectorImpl<OpFoldResult> &subviewOffsets) {
  for (unsigned d = 0; d < vi.rank; ++d) {
    int64_t blk = (d == sparseDim) ? 1 : vi.tile[d];
    Value logical =
        d == sparseDim ? sparseIndex
                       : b.create<arith::MulIOp>(
                             loc, indices[d],
                             b.create<arith::ConstantIndexOp>(loc, vi.tile[d]));
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    offset = offset ? b.create<arith::AddIOp>(loc, offset, phys) : phys;
    sizes.push_back(b.getIndexAttr(blk));
    strides.push_back(vi.strideStatic[d] == ShapedType::kDynamic
                          ? OpFoldResult(vi.strideVal[d])
                          : OpFoldResult(b.getIndexAttr(vi.strideStatic[d])));
    subviewOffsets.push_back(
        b.getIndexAttr(0)); // sparse offset overwritten by caller
  }
}

/// Gather contiguous blocks along one sparse dimension.
static Value emitContiguousGather(OpBuilder &b, Location loc,
                                  const ViewData &vi, Value base,
                                  ValueRange indices, unsigned sparseDim,
                                  RankedTensorType resultType) {
  auto bufTy = MemRefType::get(vi.tile, vi.elementType);
  Value buf = b.create<memref::AllocOp>(loc, bufTy);
  SmallVector<int64_t> blk(vi.tile.begin(), vi.tile.end());
  blk[sparseDim] = 1;
  auto layout = StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                                       vi.strideStatic);
  auto gmBlkTy = MemRefType::get(blk, vi.elementType, layout);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[sparseDim]);
  auto loop = b.create<scf::ForOp>(loc, c0, upper, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    auto ext =
        b.create<tensor::ExtractOp>(loc, indices[sparseDim], ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value sIdx = asIndex(b, loc, ext.getResult());

    Value off;
    SmallVector<OpFoldResult> sizes, strides, subOffs;
    buildContiguousBlockGeometry(b, loc, vi, indices, sparseDim, sIdx, off,
                                 sizes, strides, subOffs);
    subOffs[sparseDim] = OpFoldResult(k);
    SmallVector<OpFoldResult> subStr(vi.rank, b.getIndexAttr(1));
    Value gm = b.create<memref::ReinterpretCastOp>(
        loc, gmBlkTy, base, OpFoldResult(off), sizes, strides);
    Value sub = b.create<memref::SubViewOp>(loc, buf, subOffs, sizes, subStr);
    b.create<memref::CopyOp>(loc, gm, sub);
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());
  return b.create<bufferization::ToTensorOp>(loc, resultType, buf,
                                             /*restrict=*/true,
                                             /*writable=*/false);
}

/// Scatter contiguous blocks along one sparse dimension.
static void emitContiguousScatter(OpBuilder &b, Location loc,
                                  const ViewData &vi, Value base, Value value,
                                  ValueRange indices, unsigned sparseDim) {
  SmallVector<int64_t> blk(vi.tile.begin(), vi.tile.end());
  blk[sparseDim] = 1;
  auto layout = StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                                       vi.strideStatic);
  auto gmBlkTy = MemRefType::get(blk, vi.elementType, layout);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[sparseDim]);
  auto loop = b.create<scf::ForOp>(loc, c0, upper, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    auto ext =
        b.create<tensor::ExtractOp>(loc, indices[sparseDim], ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value sIdx = asIndex(b, loc, ext.getResult());

    Value off;
    SmallVector<OpFoldResult> sizes, strides, subOffs;
    buildContiguousBlockGeometry(b, loc, vi, indices, sparseDim, sIdx, off,
                                 sizes, strides, subOffs);
    subOffs[sparseDim] = OpFoldResult(k);
    SmallVector<OpFoldResult> subStr(vi.rank, b.getIndexAttr(1));
    Value gm = b.create<memref::ReinterpretCastOp>(
        loc, gmBlkTy, base, OpFoldResult(off), sizes, strides);
    Value slice =
        b.create<tensor::ExtractSliceOp>(loc, value, subOffs, sizes, subStr);
    auto mat =
        b.create<bufferization::MaterializeInDestinationOp>(loc, slice, gm);
    mat->setAttr("writable", b.getUnitAttr());
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());
}

/// Return whether a sparse access contains a regular contiguous block.
static bool supportsContiguousBlockTransfer(const ViewData &vi,
                                            unsigned sparseDim) {
  return llvm::any_of(llvm::enumerate(vi.strideStatic), [&](auto entry) {
    return entry.index() != sparseDim && entry.value() == 1;
  });
}

/// Gather elementwise when block DMA cannot represent the sparse access.
static Value emitDiscreteGather(OpBuilder &b, Location loc, const ViewData &vi,
                                Value base, ValueRange indices,
                                ArrayRef<int64_t> sparseDims,
                                RankedTensorType resultType) {
  Value init = b.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                         resultType.getElementType());
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<scf::ForOp> loops;
  SmallVector<Value> coords;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value iterArg = loops.empty() ? init : loops.back().getRegionIterArg(0);
    Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, c0, upper, c1, iterArg);
    if (!loops.empty())
      b.create<scf::YieldOp>(loc, loop.getResult(0));
    loops.push_back(loop);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  Value target = loops.back().getRegionIterArg(0);
  ElementAccess access =
      buildElementAccess(b, loc, vi, indices, sparseDims, coords);
  Value padding = createPaddingConstant(b, loc, vi);
  auto ifOp = b.create<scf::IfOp>(
      loc, access.inBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        Value rc = createDiscreteGmElementView(
            nested, nestedLoc, base, access.physicalOffset, vi.elementType);
        Value loaded =
            nested.create<memref::LoadOp>(nestedLoc, rc, ValueRange{c0});
        nested.create<scf::YieldOp>(nestedLoc, loaded);
      },
      [&](OpBuilder &nested, Location nestedLoc) {
        nested.create<scf::YieldOp>(nestedLoc, padding);
      });
  Value value = ifOp.getResult(0);
  Value result = b.create<tensor::InsertOp>(loc, value, target, coords);
  b.create<scf::YieldOp>(loc, result);
  b.setInsertionPointAfter(loops.front());
  return loops.front().getResult(0);
}

/// Scatter elementwise when block DMA cannot represent the sparse access.
static void emitDiscreteScatter(OpBuilder &b, Location loc, const ViewData &vi,
                                Value base, Value value, ValueRange indices,
                                ArrayRef<int64_t> sparseDims) {
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  OpBuilder::InsertionGuard guard(b);
  SmallVector<Value> coords;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, c0, upper, c1);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  ElementAccess access =
      buildElementAccess(b, loc, vi, indices, sparseDims, coords);
  b.create<scf::IfOp>(
      loc, access.inBounds, [&](OpBuilder &nested, Location nestedLoc) {
        auto element =
            nested.create<tensor::ExtractOp>(nestedLoc, value, coords);
        Value gmElement = createDiscreteGmElementView(
            nested, nestedLoc, base, access.physicalOffset, vi.elementType);
        nested.create<memref::StoreOp>(nestedLoc, element.getResult(),
                                       gmElement, ValueRange{c0});
        nested.create<scf::YieldOp>(nestedLoc);
      });
}

/// Select contiguous block transfer when possible, otherwise lower
/// elementwise.
static Value emitGatherAccess(OpBuilder &b, Location loc, const ViewData &vi,
                              Value base, ValueRange indices,
                              ArrayRef<int64_t> sparseDims,
                              RankedTensorType resultType) {
  if (sparseDims.size() == 1 &&
      supportsContiguousBlockTransfer(vi, sparseDims[0]))
    return emitContiguousGather(b, loc, vi, base, indices, sparseDims[0],
                                resultType);
  return emitDiscreteGather(b, loc, vi, base, indices, sparseDims, resultType);
}

static void emitScatterAccess(OpBuilder &b, Location loc, const ViewData &vi,
                              Value base, Value value, ValueRange indices,
                              ArrayRef<int64_t> sparseDims) {
  if (sparseDims.size() == 1 &&
      supportsContiguousBlockTransfer(vi, sparseDims[0]))
    emitContiguousScatter(b, loc, vi, base, value, indices, sparseDims[0]);
  else
    emitDiscreteScatter(b, loc, vi, base, value, indices, sparseDims);
}

//===----------------------------------------------------------------------===//
// View load/store lowering
//===----------------------------------------------------------------------===//

static LogicalResult lowerViewLoad(tv::ViewLoadOp load,
                                   PatternRewriter &rewriter) {
  ViewData vi = parseView(load.getView());
  if (!vi.ok || load.getIndices().size() != vi.rank)
    return failure();

  rewriter.setInsertionPoint(load);
  OpBuilder &b = rewriter;
  Location loc = load.getLoc();
  auto tensorTy = cast<RankedTensorType>(load.getResult().getType());

  if (vi.isGatherScatter()) {
    for (unsigned d = 0; d < vi.rank; ++d) {
      bool isSparse =
          llvm::is_contained(vi.sparseDims, static_cast<int64_t>(d));
      if (isSparse != isa<RankedTensorType>(load.getIndices()[d].getType()))
        return failure();
    }
  }

  std::optional<PhysicalTransferLayout> transferLayout;
  if (!vi.isGatherScatter()) {
    transferLayout = buildPhysicalTransferLayout(vi);
    if (!transferLayout)
      return failure();
  }

  vi.base = materializeBase(b, loc, vi);
  Value accessInBounds =
      buildAccessInBoundsCondition(b, loc, vi, load.getIndices());
  auto guardedLoad = b.create<scf::IfOp>(
      loc, accessInBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        Value result;
        if (vi.isGatherScatter()) {
          result = emitGatherAccess(nested, nestedLoc, vi, vi.base,
                                    load.getIndices(), vi.sparseDims, tensorTy);
        } else {
          bool collapsed = transferLayout->collapses(vi.rank);
          auto localType = MemRefType::get(vi.tile, vi.elementType);
          auto transferLocalType =
              MemRefType::get(transferLayout->shape, vi.elementType);
          Value buffer = nested.create<memref::AllocOp>(
              nestedLoc, collapsed ? transferLocalType : localType);
          Value offset =
              buildTileOriginOffset(nested, nestedLoc, vi, load.getIndices());
          Value gm =
              createGmTileView(nested, nestedLoc, vi, offset, &*transferLayout);
          nested.create<memref::CopyOp>(nestedLoc, gm, buffer);

          Value tensorBuffer = buffer;
          if (collapsed)
            tensorBuffer = nested.create<memref::ExpandShapeOp>(
                nestedLoc, localType, buffer, transferLayout->reassociation);
          result = nested.create<bufferization::ToTensorOp>(
              nestedLoc, tensorTy, tensorBuffer,
              /*restrict=*/true, /*writable=*/false);
        }
        nested.create<scf::YieldOp>(nestedLoc, result);
      },
      [&](OpBuilder &nested, Location nestedLoc) {
        Value result =
            emitDiscreteGather(nested, nestedLoc, vi, vi.base,
                               load.getIndices(), vi.sparseDims, tensorTy);
        nested.create<scf::YieldOp>(nestedLoc, result);
      });
  rewriter.replaceOp(load, guardedLoad.getResults());
  return success();
}

static LogicalResult lowerViewStore(tv::ViewStoreOp store,
                                    PatternRewriter &rewriter) {
  ViewData vi = parseView(store.getView());
  if (!vi.ok || store.getIndices().size() != vi.rank)
    return failure();

  rewriter.setInsertionPoint(store);
  OpBuilder &b = rewriter;
  Location loc = store.getLoc();

  if (vi.isGatherScatter()) {
    for (unsigned d = 0; d < vi.rank; ++d) {
      bool isSparse =
          llvm::is_contained(vi.sparseDims, static_cast<int64_t>(d));
      if (isSparse != isa<RankedTensorType>(store.getIndices()[d].getType()))
        return failure();
    }
  }

  vi.base = materializeBase(b, loc, vi);
  Value accessInBounds =
      buildAccessInBoundsCondition(b, loc, vi, store.getIndices());
  b.create<scf::IfOp>(
      loc, accessInBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        if (vi.isGatherScatter())
          emitScatterAccess(nested, nestedLoc, vi, vi.base, store.getValue(),
                            store.getIndices(), vi.sparseDims);
        else {
          Value offset =
              buildTileOriginOffset(nested, nestedLoc, vi, store.getIndices());
          Value gm = createGmTileView(nested, nestedLoc, vi, offset,
                                      /*transferLayout=*/nullptr);
          auto materialize =
              nested.create<bufferization::MaterializeInDestinationOp>(
                  nestedLoc, store.getValue(), gm);
          materialize->setAttr("writable", nested.getUnitAttr());
        }
        nested.create<scf::YieldOp>(nestedLoc);
      },
      [&](OpBuilder &nested, Location nestedLoc) {
        emitDiscreteScatter(nested, nestedLoc, vi, vi.base, store.getValue(),
                            store.getIndices(), vi.sparseDims);
        nested.create<scf::YieldOp>(nestedLoc);
      });

  rewriter.eraseOp(store);
  return success();
}

//===----------------------------------------------------------------------===//
// Discrete pointer accesses
//===----------------------------------------------------------------------===//

static LogicalResult lowerPtrLoad(tv::PtrLoadOp op, PatternRewriter &rewriter) {
  Value pointers = op.getPointers();
  auto pointersType = dyn_cast<RankedTensorType>(pointers.getType());
  if (!pointersType ||
      op.getIndices().size() != static_cast<size_t>(pointersType.getRank()))
    return failure();
  auto resTy = cast<RankedTensorType>(op.getResult().getType());
  int64_t n = resTy.getShape()[0];
  if (ShapedType::isDynamic(n))
    return failure();

  rewriter.setInsertionPoint(op);
  OpBuilder &b = rewriter;
  Location loc = op.getLoc();
  Value init =
      b.create<tensor::EmptyOp>(loc, resTy.getShape(), resTy.getElementType());
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value ub = b.create<arith::ConstantIndexOp>(loc, n);
  auto loop = b.create<scf::ForOp>(loc, c0, ub, c1, ValueRange{init});
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    SmallVector<Value> coordinates;
    for (Value index : op.getIndices())
      coordinates.push_back(
          b.create<tensor::ExtractOp>(loc, index, ValueRange{k}));
    Value pointer = b.create<tensor::ExtractOp>(loc, pointers, coordinates);
    auto load = b.create<triton::LoadOp>(
        loc, pointer, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, /*isVolatile=*/false);
    load->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value value = load.getResult();
    Value result = b.create<tensor::InsertOp>(
        loc, value, loop.getRegionIterArg(0), ValueRange{k});
    auto yield = b.create<scf::YieldOp>(loc, result);
    yield->setAttr("DiscreteMemAccess", b.getUnitAttr());
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());

  rewriter.replaceOp(op, loop.getResult(0));
  return success();
}

static LogicalResult lowerPtrStore(tv::PtrStoreOp op,
                                   PatternRewriter &rewriter) {
  Value pointers = op.getPointers();
  auto pointersType = dyn_cast<RankedTensorType>(pointers.getType());
  if (!pointersType ||
      op.getIndices().size() != static_cast<size_t>(pointersType.getRank()))
    return failure();
  Value value = op.getValue();
  auto valTy = cast<RankedTensorType>(value.getType());
  int64_t n = valTy.getShape()[0];
  if (ShapedType::isDynamic(n))
    return failure();

  rewriter.setInsertionPoint(op);
  OpBuilder &b = rewriter;
  Location loc = op.getLoc();
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value ub = b.create<arith::ConstantIndexOp>(loc, n);
  auto loop = b.create<scf::ForOp>(loc, c0, ub, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    SmallVector<Value> coordinates;
    for (Value index : op.getIndices())
      coordinates.push_back(
          b.create<tensor::ExtractOp>(loc, index, ValueRange{k}));
    Value pointer = b.create<tensor::ExtractOp>(loc, pointers, coordinates);
    Value element = b.create<tensor::ExtractOp>(loc, value, ValueRange{k});
    auto store = b.create<triton::StoreOp>(loc, pointer, element,
                                           triton::CacheModifier::NONE,
                                           triton::EvictionPolicy::NORMAL);
    store->setAttr("DiscreteMemAccess", b.getUnitAttr());
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());

  rewriter.eraseOp(op);
  return success();
}

struct ViewLoadOpConversion : public OpRewritePattern<tv::ViewLoadOp> {
  using OpRewritePattern<tv::ViewLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tv::ViewLoadOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(lowerViewLoad(op, rewriter)))
      return rewriter.notifyMatchFailure(op, "unsupported TensorView load");
    return success();
  }
};

struct ViewStoreOpConversion : public OpRewritePattern<tv::ViewStoreOp> {
  using OpRewritePattern<tv::ViewStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tv::ViewStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(lowerViewStore(op, rewriter)))
      return rewriter.notifyMatchFailure(op, "unsupported TensorView store");
    return success();
  }
};

struct PtrLoadOpConversion : public OpRewritePattern<tv::PtrLoadOp> {
  using OpRewritePattern<tv::PtrLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tv::PtrLoadOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(lowerPtrLoad(op, rewriter)))
      return rewriter.notifyMatchFailure(op, "unsupported pointer load");
    return success();
  }
};

struct PtrStoreOpConversion : public OpRewritePattern<tv::PtrStoreOp> {
  using OpRewritePattern<tv::PtrStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tv::PtrStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(lowerPtrStore(op, rewriter)))
      return rewriter.notifyMatchFailure(op, "unsupported pointer store");
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Function-argument rewrite
//===----------------------------------------------------------------------===//

static void rewriteFuncPtrArgs(triton::FuncOp func) {
  auto ptrToMemref = [&](Type t) -> Type {
    if (auto p = dyn_cast<tv::PtrType>(t))
      return MemRefType::get({ShapedType::kDynamic}, p.getPointeeType());
    return t;
  };

  auto funcTy = func.getFunctionType();
  bool changed = false;
  SmallVector<Type> inputs;
  for (Type t : funcTy.getInputs()) {
    Type nt = ptrToMemref(t);
    changed |= (nt != t);
    inputs.push_back(nt);
  }
  if (!changed)
    return;

  func.setFunctionType(
      FunctionType::get(func.getContext(), inputs, funcTy.getResults()));
  if (!func.empty())
    for (BlockArgument arg : func.front().getArguments())
      if (isa<tv::PtrType>(arg.getType()))
        arg.setType(ptrToMemref(arg.getType()));
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TensorViewLoweringPass
    : public mlir::triton::impl::TensorViewLoweringBase<
          TensorViewLoweringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](triton::FuncOp func) { rewriteFuncPtrArgs(func); });

    RewritePatternSet patterns(&getContext());
    patterns.add<ViewLoadOpConversion, ViewStoreOpConversion,
                 PtrLoadOpConversion, PtrStoreOpConversion>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();

    WalkResult unloweredAccess = module.walk([&](Operation *operation) {
      if (!isa<tv::ViewLoadOp, tv::ViewStoreOp, tv::PtrLoadOp, tv::PtrStoreOp>(
              operation))
        return WalkResult::advance();
      operation->emitError("TensorViewLowering could not lower this access");
      return WalkResult::interrupt();
    });
    if (unloweredAccess.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTensorViewLoweringPass() {
  return std::make_unique<TensorViewLoweringPass>();
}
