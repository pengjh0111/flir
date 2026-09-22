//===- TensorViewLowering.cpp - tv access ops -> generic memref
//-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower TensorView operations to portable linalg, memref, bufferization, scf,
// and tensor operations. Address spaces and DMA selection remain backend
// concerns.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TensorViewLowering/Passes.h"

#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
#include "llvm/ADT/STLFunctionalExtras.h"
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

/// Physical layout of one GM-to-local transfer. Unit and fixed sparse
/// dimensions contribute to offsets but do not increase the transfer rank.
struct PhysicalTransferLayout {
  SmallVector<unsigned> logicalDims;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strideStatic;
  SmallVector<int64_t> localStrideStatic;
  SmallVector<ReassociationIndices> reassociation;

  bool collapses(unsigned logicalRank) const {
    return logicalDims.size() != logicalRank;
  }
};

static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (unsigned d = shape.size(); d > 0; --d) {
    strides[d - 1] = stride;
    stride *= shape[d - 1];
  }
  return strides;
}

static std::optional<PhysicalTransferLayout>
buildPhysicalTransferLayout(const ViewData &view, ArrayRef<int64_t> fixedDims) {
  PhysicalTransferLayout layout;
  SmallVector<int64_t> localStrides = computeRowMajorStrides(view.tile);

  for (unsigned d = 0; d < view.rank; ++d) {
    if (view.tile[d] == 1 ||
        llvm::is_contained(fixedDims, static_cast<int64_t>(d)))
      continue;
    layout.logicalDims.push_back(d);
    layout.shape.push_back(view.tile[d]);
    layout.strideStatic.push_back(view.strideStatic[d]);
    layout.localStrideStatic.push_back(localStrides[d]);
  }

  // A view without sliced dimensions still represents a one-element block.
  if (layout.logicalDims.empty() && fixedDims.empty() && view.rank != 0) {
    unsigned d = view.rank - 1;
    layout.logicalDims.push_back(d);
    layout.shape.push_back(view.tile[d]);
    layout.strideStatic.push_back(view.strideStatic[d]);
    layout.localStrideStatic.push_back(localStrides[d]);
  }

  if (layout.logicalDims.empty())
    return std::nullopt;

  if (!fixedDims.empty())
    return layout;

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

struct BlockTransferGeometry {
  Value gmOffset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> localOffsets;
};

static BlockTransferGeometry buildBlockTransferGeometry(
    OpBuilder &b, Location loc, const ViewData &vi, ValueRange indices,
    ArrayRef<Value> sparseCoordinates, const PhysicalTransferLayout &layout,
    bool boundary) {
  BlockTransferGeometry geometry;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value hasElements;
  if (boundary)
    hasElements = b.create<arith::ConstantIntOp>(loc, 1, 1);

  for (unsigned d = 0; d < vi.rank; ++d) {
    bool sparse = static_cast<bool>(sparseCoordinates[d]);
    Value origin;
    int64_t blockSize;
    if (sparse) {
      auto extracted = b.create<tensor::ExtractOp>(
          loc, indices[d], ValueRange{sparseCoordinates[d]});
      extracted->setAttr("DiscreteMemAccess", b.getUnitAttr());
      origin = asIndex(b, loc, extracted.getResult());
      blockSize = 1;
    } else {
      Value traversal = b.create<arith::ConstantIndexOp>(loc, vi.traversal[d]);
      origin = b.create<arith::MulIOp>(loc, indices[d], traversal);
      blockSize = vi.tile[d];
    }

    Value begin = origin;
    OpFoldResult size = b.getIndexAttr(blockSize);
    OpFoldResult localOffset = b.getIndexAttr(0);
    if (boundary) {
      Value block = b.create<arith::ConstantIndexOp>(loc, blockSize);
      begin = b.create<arith::MinSIOp>(
          loc, b.create<arith::MaxSIOp>(loc, origin, zero), vi.sizeVal[d]);
      Value end = b.create<arith::MinSIOp>(
          loc, b.create<arith::AddIOp>(loc, origin, block), vi.sizeVal[d]);
      Value clippedSize = b.create<arith::MaxSIOp>(
          loc, b.create<arith::SubIOp>(loc, end, begin), zero);
      Value dimensionHasElements = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, clippedSize, zero);
      hasElements =
          b.create<arith::AndIOp>(loc, hasElements, dimensionHasElements);
      size = clippedSize;
      Value offset = b.create<arith::MaxSIOp>(
          loc, b.create<arith::SubIOp>(loc, begin, origin), zero);
      localOffset = b.create<arith::MinSIOp>(loc, offset, block).getResult();
    }

    Value physical = b.create<arith::MulIOp>(loc, begin, vi.strideVal[d]);
    geometry.gmOffset =
        geometry.gmOffset
            ? b.create<arith::AddIOp>(loc, geometry.gmOffset, physical)
            : physical;

    if (llvm::is_contained(layout.logicalDims, d)) {
      geometry.sizes.push_back(size);
      geometry.localOffsets.push_back(localOffset);
    }
  }

  if (boundary) {
    Value firstSize =
        getValueOrCreateConstantIndexOp(b, loc, geometry.sizes.front());
    geometry.sizes.front() =
        b.create<arith::SelectOp>(loc, hasElements, firstSize, zero)
            .getResult();
  }
  return geometry;
}

static void
emitSparseCoordinateLoops(OpBuilder &b, Location loc, const ViewData &vi,
                          unsigned sparsePosition,
                          SmallVectorImpl<Value> &coordinates,
                          llvm::function_ref<void(ArrayRef<Value>)> emitBody) {
  if (sparsePosition == vi.sparseDims.size()) {
    emitBody(coordinates);
    return;
  }

  unsigned dim = vi.sparseDims[sparsePosition];
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[dim]);
  auto loop = b.create<scf::ForOp>(loc, zero, upper, one);
  loop->setAttr("hivm.parallel_loop", b.getUnitAttr());
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(loop.getBody());
  coordinates[dim] = loop.getInductionVar();
  emitSparseCoordinateLoops(b, loc, vi, sparsePosition + 1, coordinates,
                            emitBody);
  coordinates[dim] = Value();
}

static Value createLocalTransferView(OpBuilder &b, Location loc, Value buffer,
                                     Value offset,
                                     const PhysicalTransferLayout &layout,
                                     ArrayRef<int64_t> localStrides,
                                     Type elementType) {
  SmallVector<OpFoldResult> sizes, strides;
  for (auto [size, stride] : llvm::zip_equal(layout.shape, localStrides)) {
    sizes.push_back(b.getIndexAttr(size));
    strides.push_back(b.getIndexAttr(stride));
  }
  auto localLayout = StridedLayoutAttr::get(b.getContext(),
                                            ShapedType::kDynamic, localStrides);
  auto type = MemRefType::get(layout.shape, elementType, localLayout);
  return b.create<memref::ReinterpretCastOp>(
      loc, type, buffer, OpFoldResult(offset), sizes, strides);
}

static void emitBlockCopy(OpBuilder &b, Location loc, const ViewData &vi,
                          Value localBuffer, Value localOffset,
                          ArrayRef<int64_t> localStrides, ValueRange indices,
                          ArrayRef<Value> sparseCoordinates,
                          const PhysicalTransferLayout &layout, bool boundary,
                          bool store) {
  BlockTransferGeometry geometry = buildBlockTransferGeometry(
      b, loc, vi, indices, sparseCoordinates, layout, boundary);

  Value gm = createGmTileView(b, loc, vi, geometry.gmOffset, &layout);
  Value local = createLocalTransferView(b, loc, localBuffer, localOffset,
                                        layout, localStrides, vi.elementType);
  if (boundary) {
    SmallVector<OpFoldResult> zeros(layout.shape.size(), b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(layout.shape.size(), b.getIndexAttr(1));
    gm = b.create<memref::SubViewOp>(loc, gm, zeros, geometry.sizes, strides);
    local = b.create<memref::SubViewOp>(loc, local, geometry.localOffsets,
                                        geometry.sizes, strides);
  }
  if (store)
    b.create<memref::CopyOp>(loc, local, gm);
  else
    b.create<memref::CopyOp>(loc, gm, local);
}

struct SparseResultLayout {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> sliceShape;
  unsigned splitDim;
  int64_t splitSize;
  int64_t chunksAtSplit;
};

static std::optional<SparseResultLayout>
buildSparseResultLayout(const ViewData &vi, RankedTensorType resultType) {
  if (!resultType.hasStaticShape())
    return std::nullopt;

  bool reachedDenseDimensions = false;
  int64_t previousSparseDim = -1;
  int64_t sparseSlices = 1;
  for (int64_t dim : vi.sparseDims) {
    if (dim <= previousSparseDim || dim < 0 ||
        dim >= static_cast<int64_t>(vi.rank))
      return std::nullopt;
    previousSparseDim = dim;
    sparseSlices *= vi.tile[dim];
  }
  for (unsigned dim = 0; dim < vi.rank; ++dim) {
    bool sparse = llvm::is_contained(vi.sparseDims, static_cast<int64_t>(dim));
    if (sparse && reachedDenseDimensions)
      return std::nullopt;
    if (!sparse && vi.tile[dim] != 1)
      reachedDenseDimensions = true;
  }

  int64_t sourceElements = 1;
  for (int64_t size : vi.tile)
    sourceElements *= size;
  if (sourceElements != resultType.getNumElements())
    return std::nullopt;

  int64_t sliceElements = sourceElements / sparseSlices;
  ArrayRef<int64_t> resultShape = resultType.getShape();
  int64_t suffixElements = 1;
  for (unsigned dim = resultShape.size(); dim > 0; --dim) {
    unsigned candidate = dim - 1;
    if (sliceElements % suffixElements == 0) {
      int64_t splitSize = sliceElements / suffixElements;
      if (splitSize <= resultShape[candidate] &&
          resultShape[candidate] % splitSize == 0) {
        SparseResultLayout layout;
        layout.shape.assign(resultShape.begin(), resultShape.end());
        layout.sliceShape.assign(resultShape.size(), 1);
        layout.splitDim = candidate;
        layout.splitSize = splitSize;
        layout.chunksAtSplit = resultShape[candidate] / splitSize;
        for (unsigned trailing = candidate + 1; trailing < resultShape.size();
             ++trailing)
          layout.sliceShape[trailing] = resultShape[trailing];
        layout.sliceShape[candidate] = splitSize;
        return layout;
      }
    }
    suffixElements *= resultShape[candidate];
  }
  return std::nullopt;
}

static Value buildSparseSliceOffset(OpBuilder &b, Location loc,
                                    const ViewData &vi,
                                    ArrayRef<Value> sparseCoordinates) {
  SmallVector<int64_t> strides = computeRowMajorStrides(vi.tile);
  Value offset = b.create<arith::ConstantIndexOp>(loc, 0);
  for (int64_t dim : vi.sparseDims) {
    Value contribution = b.create<arith::MulIOp>(
        loc, sparseCoordinates[dim],
        b.create<arith::ConstantIndexOp>(loc, strides[dim]));
    offset = b.create<arith::AddIOp>(loc, offset, contribution);
  }
  return offset;
}

static Value emitSparseBlockLoad(OpBuilder &b, Location loc, const ViewData &vi,
                                 ValueRange indices,
                                 const PhysicalTransferLayout &layout,
                                 bool boundary, unsigned sparsePosition,
                                 SmallVectorImpl<Value> &sparseCoordinates,
                                 Value result,
                                 const SparseResultLayout *resultLayout) {
  if (sparsePosition != vi.sparseDims.size()) {
    unsigned dim = vi.sparseDims[sparsePosition];
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[dim]);
    auto loop = b.create<scf::ForOp>(loc, zero, upper, one, ValueRange{result});
    loop->setAttr("hivm.parallel_loop", b.getUnitAttr());
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      sparseCoordinates[dim] = loop.getInductionVar();
      Value next = emitSparseBlockLoad(b, loc, vi, indices, layout, boundary,
                                       sparsePosition + 1, sparseCoordinates,
                                       loop.getRegionIterArg(0), resultLayout);
      b.create<scf::YieldOp>(loc, next);
    }
    sparseCoordinates[dim] = Value();
    return loop.getResult(0);
  }

  SmallVector<int64_t> sliceShape;
  if (resultLayout) {
    sliceShape = resultLayout->sliceShape;
  } else {
    sliceShape.assign(vi.tile.begin(), vi.tile.end());
    for (int64_t dim : vi.sparseDims)
      sliceShape[dim] = 1;
  }
  auto sliceType = MemRefType::get(sliceShape, vi.elementType);
  Value slice = b.create<memref::AllocOp>(loc, sliceType);
  if (boundary) {
    Value padding = createPaddingConstant(b, loc, vi);
    b.create<linalg::FillOp>(loc, ValueRange{padding}, ValueRange{slice});
  }

  SmallVector<int64_t> transferStrides;
  if (resultLayout) {
    transferStrides = computeRowMajorStrides(layout.shape);
  } else {
    SmallVector<int64_t> sliceStrides = computeRowMajorStrides(sliceShape);
    for (unsigned dim : layout.logicalDims)
      transferStrides.push_back(sliceStrides[dim]);
  }
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  emitBlockCopy(b, loc, vi, slice, zero, transferStrides, indices,
                sparseCoordinates, layout, boundary, /*store=*/false);

  auto sliceTensorType = RankedTensorType::get(sliceShape, vi.elementType);
  Value sliceTensor = b.create<bufferization::ToTensorOp>(
      loc, sliceTensorType, slice, /*restrict=*/true, /*writable=*/false);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  if (!resultLayout) {
    for (unsigned d = 0; d < vi.rank; ++d)
      offsets.push_back(sparseCoordinates[d]
                            ? OpFoldResult(sparseCoordinates[d])
                            : OpFoldResult(b.getIndexAttr(0)));
  } else {
    Value ordinal = b.create<arith::ConstantIndexOp>(loc, 0);
    for (int64_t dim : vi.sparseDims) {
      ordinal = b.create<arith::MulIOp>(
          loc, ordinal, b.create<arith::ConstantIndexOp>(loc, vi.tile[dim]));
      ordinal = b.create<arith::AddIOp>(loc, ordinal, sparseCoordinates[dim]);
    }

    offsets.assign(resultLayout->shape.size(), b.getIndexAttr(0));
    Value remaining = ordinal;
    for (unsigned dim = resultLayout->splitDim + 1; dim > 0; --dim) {
      unsigned resultDim = dim - 1;
      int64_t chunks = resultDim == resultLayout->splitDim
                           ? resultLayout->chunksAtSplit
                           : resultLayout->shape[resultDim];
      Value coordinate = remaining;
      if (resultDim != 0) {
        Value divisor = b.create<arith::ConstantIndexOp>(loc, chunks);
        coordinate = b.create<arith::RemSIOp>(loc, remaining, divisor);
        remaining = b.create<arith::DivSIOp>(loc, remaining, divisor);
      }
      if (resultDim == resultLayout->splitDim)
        coordinate = b.create<arith::MulIOp>(
            loc, coordinate,
            b.create<arith::ConstantIndexOp>(loc, resultLayout->splitSize));
      offsets[resultDim] = coordinate;
    }
  }
  for (int64_t size : sliceShape) {
    sizes.push_back(b.getIndexAttr(size));
    strides.push_back(b.getIndexAttr(1));
  }
  return b.create<tensor::InsertSliceOp>(loc, sliceTensor, result, offsets,
                                         sizes, strides);
}

static Value emitBlockLoad(OpBuilder &b, Location loc, const ViewData &vi,
                           ValueRange indices,
                           const PhysicalTransferLayout &layout,
                           RankedTensorType resultType, bool boundary,
                           const SparseResultLayout *resultLayout = nullptr) {
  bool sparse = vi.isGatherScatter();
  if (sparse) {
    Value result = b.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                             resultType.getElementType());
    SmallVector<Value> sparseCoordinates(vi.rank);
    return emitSparseBlockLoad(b, loc, vi, indices, layout, boundary,
                               /*sparsePosition=*/0, sparseCoordinates, result,
                               resultLayout);
  }

  auto bufferType = MemRefType::get(layout.shape, vi.elementType);
  Value buffer = b.create<memref::AllocOp>(loc, bufferType);
  if (boundary) {
    Value padding = createPaddingConstant(b, loc, vi);
    b.create<linalg::FillOp>(loc, ValueRange{padding}, ValueRange{buffer});
  }

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> sparseCoordinates(vi.rank);
  emitBlockCopy(b, loc, vi, buffer, zero, layout.localStrideStatic, indices,
                sparseCoordinates, layout, boundary, /*store=*/false);

  Value tensorBuffer = buffer;
  if (layout.collapses(vi.rank)) {
    auto logicalType = MemRefType::get(vi.tile, vi.elementType);
    tensorBuffer = b.create<memref::ExpandShapeOp>(loc, logicalType, buffer,
                                                   layout.reassociation);
  }
  return b.create<bufferization::ToTensorOp>(loc, resultType, tensorBuffer,
                                             /*restrict=*/true,
                                             /*writable=*/false);
}

static void emitBlockStore(OpBuilder &b, Location loc, const ViewData &vi,
                           Value value, ValueRange indices,
                           const PhysicalTransferLayout &layout,
                           bool boundary) {
  bool sparse = vi.isGatherScatter();
  Value physicalValue = value;
  if (!sparse && layout.collapses(vi.rank)) {
    auto type = RankedTensorType::get(layout.shape, vi.elementType);
    physicalValue = b.create<tensor::CollapseShapeOp>(loc, type, value,
                                                      layout.reassociation);
  }
  auto bufferType = MemRefType::get(sparse ? ArrayRef<int64_t>(vi.tile)
                                           : ArrayRef<int64_t>(layout.shape),
                                    vi.elementType);
  Value buffer =
#if LLVM_VERSION_MAJOR < 22
      b.create<bufferization::ToMemrefOp>(loc, bufferType, physicalValue);
#else
      b.create<bufferization::ToBufferOp>(loc, bufferType, physicalValue);
#endif

  SmallVector<Value> sparseCoordinates(vi.rank);
  emitSparseCoordinateLoops(
      b, loc, vi, 0, sparseCoordinates, [&](ArrayRef<Value> coordinates) {
        Value offset = buildSparseSliceOffset(b, loc, vi, coordinates);
        emitBlockCopy(b, loc, vi, buffer, offset, layout.localStrideStatic,
                      indices, coordinates, layout, boundary, /*store=*/true);
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
        Value element =
            nested.create<tensor::ExtractOp>(nestedLoc, value, coords);
        Value gmElement = createDiscreteGmElementView(
            nested, nestedLoc, base, access.physicalOffset, vi.elementType);
        Value empty = nested.create<tensor::EmptyOp>(
            nestedLoc, ArrayRef<int64_t>{1}, vi.elementType);
        Value inserted = nested.create<tensor::InsertOp>(nestedLoc, element,
                                                         empty, ValueRange{c0});
        auto materialize =
            nested.create<bufferization::MaterializeInDestinationOp>(
                nestedLoc, inserted, gmElement);
        materialize->setAttr("writable", nested.getUnitAttr());
        nested.create<scf::YieldOp>(nestedLoc);
      });
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

  std::optional<PhysicalTransferLayout> transferLayout =
      buildPhysicalTransferLayout(vi, vi.sparseDims);
  if (!vi.isGatherScatter() && !transferLayout)
    return failure();

  triton::ReshapeOp reshape;
  std::optional<SparseResultLayout> resultLayout;
  if (vi.isGatherScatter() && transferLayout && load.getResult().hasOneUse()) {
    reshape = dyn_cast<triton::ReshapeOp>(*load.getResult().user_begin());
    if (reshape && !reshape->hasAttr("allow_reorder")) {
      auto reshapeType =
          dyn_cast<RankedTensorType>(reshape.getResult().getType());
      if (reshapeType)
        resultLayout = buildSparseResultLayout(vi, reshapeType);
    }
    if (!resultLayout)
      reshape = nullptr;
    else
      tensorTy = cast<RankedTensorType>(reshape.getResult().getType());
  }

  vi.base = materializeBase(b, loc, vi);
  Value accessInBounds =
      buildAccessInBoundsCondition(b, loc, vi, load.getIndices());
  auto guardedLoad = b.create<scf::IfOp>(
      loc, accessInBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        Value result;
        if (vi.isGatherScatter()) {
          result =
              transferLayout
                  ? emitBlockLoad(nested, nestedLoc, vi, load.getIndices(),
                                  *transferLayout, tensorTy, /*boundary=*/false,
                                  resultLayout ? &*resultLayout : nullptr)
                  : emitDiscreteGather(nested, nestedLoc, vi, vi.base,
                                       load.getIndices(), vi.sparseDims,
                                       tensorTy);
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
            transferLayout
                ? emitBlockLoad(nested, nestedLoc, vi, load.getIndices(),
                                *transferLayout, tensorTy, /*boundary=*/true,
                                resultLayout ? &*resultLayout : nullptr)
                : emitDiscreteGather(nested, nestedLoc, vi, vi.base,
                                     load.getIndices(), vi.sparseDims,
                                     tensorTy);
        nested.create<scf::YieldOp>(nestedLoc, result);
      });
  if (reshape) {
    rewriter.replaceOp(reshape, guardedLoad.getResults());
    rewriter.eraseOp(load);
  } else {
    rewriter.replaceOp(load, guardedLoad.getResults());
  }
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

  std::optional<PhysicalTransferLayout> transferLayout =
      buildPhysicalTransferLayout(vi, vi.sparseDims);
  if (!vi.isGatherScatter() && !transferLayout)
    return failure();

  vi.base = materializeBase(b, loc, vi);
  Value accessInBounds =
      buildAccessInBoundsCondition(b, loc, vi, store.getIndices());
  b.create<scf::IfOp>(
      loc, accessInBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        if (vi.isGatherScatter()) {
          if (transferLayout)
            emitBlockStore(nested, nestedLoc, vi, store.getValue(),
                           store.getIndices(), *transferLayout,
                           /*boundary=*/false);
          else
            emitDiscreteScatter(nested, nestedLoc, vi, vi.base,
                                store.getValue(), store.getIndices(),
                                vi.sparseDims);
        } else {
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
        if (transferLayout)
          emitBlockStore(nested, nestedLoc, vi, store.getValue(),
                         store.getIndices(), *transferLayout,
                         /*boundary=*/true);
        else
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
