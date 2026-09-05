//===- TensorViewLowering.cpp - tv access ops -> generic memref -------------===//
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

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
  Value base;                        // flat memref<?xT> (plain, no address space)
  Type elementType;
  unsigned rank = 0;
  SmallVector<int64_t> tile;         // per-dim tile size
  SmallVector<int64_t> traversal;    // per-dim traversal stride (== tile for partition)
  SmallVector<int64_t> strideStatic; // per-dim element stride, or kDynamic
  SmallVector<Value> strideVal;      // per-dim element stride SSA (index)
  SmallVector<Value> n;              // per-dim extent (index) from make_tensor_view
  SmallVector<int64_t> sparseDims;   // gather/scatter dimensions (one or more)
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
  if (isa<tv::GatherScatterViewAttr>(encoding) && vi.sparseDims.empty())
    return vi;

  // Follow the encoded view to its base view.
  Operation *viewOp = viewVal.getDefiningOp();
  Value baseView;
  if (auto p = dyn_cast_or_null<tv::MakePartitionViewOp>(viewOp))
    baseView = p.getSource();
  else if (auto s = dyn_cast_or_null<tv::MakeStridedViewOp>(viewOp))
    baseView = s.getSource();
  else if (auto g = dyn_cast_or_null<tv::MakeGatherScatterViewOp>(viewOp))
    baseView = g.getSource();
  else
    return vi;

  auto mtv = baseView.getDefiningOp<tv::MakeTensorViewOp>();
  if (!mtv)
    return vi;
  auto baseTy = dyn_cast<tv::TensorViewType>(mtv.getResult().getType());
  if (!baseTy)
    return vi;
  // Read the source untyped after function arguments have become memrefs.
  Value base = mtv->getOperand(0);
  // Resolve bridge casts introduced for non-block-argument pointer bases.
  auto bridgeCast = base.getDefiningOp<UnrealizedConversionCastOp>();
  if (bridgeCast && bridgeCast.getNumOperands() == 1 &&
      isa<triton::PointerType>(bridgeCast.getOperand(0).getType())) {
    auto ptrTy = cast<triton::PointerType>(bridgeCast.getOperand(0).getType());
    auto memrefTy = MemRefType::get({ShapedType::kDynamic}, ptrTy.getPointeeType());
    OpBuilder b(mtv);
    Location loc = mtv.getLoc();
    // Peel the complete addptr chain so native lowering cannot convert an
    // interior pointer independently.
    Value chainOffset;
    Value root = bridgeCast.getOperand(0);
    while (auto addptr = root.getDefiningOp<triton::AddPtrOp>()) {
      if (isa<RankedTensorType>(addptr.getOffset().getType()))
        break;
      Value offsetIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
                                                      addptr.getOffset());
      chainOffset = chainOffset
                       ? b.create<arith::AddIOp>(loc, chainOffset, offsetIdx)
                             .getResult()
                       : offsetIdx;
      root = addptr.getPtr();
    }
    Value rootBase =
        b.create<UnrealizedConversionCastOp>(loc, memrefTy, root)
            .getResult(0);
    if (chainOffset) {
      // Downstream users reinterpret this shifted scalar view with real sizes.
      auto layout = StridedLayoutAttr::get(b.getContext(),
                                           /*offset=*/ShapedType::kDynamic, {1});
      auto viewTy = MemRefType::get({1}, ptrTy.getPointeeType(), layout);
      base = b.create<memref::ReinterpretCastOp>(
                  loc, viewTy, rootBase, OpFoldResult(chainOffset),
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1)})
                 .getResult();
    } else {
      base = rootBase;
    }
  }
  if (!isa<MemRefType>(base.getType()))
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
  vi.strideStatic.assign(baseTy.getStrides().begin(), baseTy.getStrides().end());
  vi.strideVal.assign(mtv.getStrides().begin(), mtv.getStrides().end());
  vi.n.assign(mtv.getSizes().begin(), mtv.getSizes().end());
  vi.ok = true;
  return vi;
}

/// Drop tile-size-one dimensions from DMA shapes, retaining at least rank one.
static SmallVector<unsigned> getNonPinnedDims(ArrayRef<int64_t> tile) {
  SmallVector<unsigned> kept;
  for (unsigned d = 0; d < tile.size(); ++d)
    if (tile[d] != 1)
      kept.push_back(d);
  if (kept.empty() && !tile.empty())
    kept.push_back(tile.size() - 1);
  return kept;
}

/// Build reassociation groups for restoring collapsed pinned dimensions.
static SmallVector<ReassociationIndices>
buildPinnedDimReassociation(ArrayRef<unsigned> kept, unsigned rank) {
  SmallVector<ReassociationIndices> reassoc;
  unsigned prevEnd = 0;
  for (unsigned j = 0; j < kept.size(); ++j) {
    unsigned end = (j + 1 == kept.size()) ? rank : kept[j] + 1;
    ReassociationIndices group;
    for (unsigned d = prevEnd; d < end; ++d)
      group.push_back(d);
    reassoc.push_back(group);
    prevEnd = end;
  }
  return reassoc;
}

/// Reinterpret the base as a GM tile, optionally collapsing pinned dimensions
/// for backend DMA rank limits.
static Value createGmTile(OpBuilder &b, Location loc, const ViewData &vi,
                          Value off, bool collapse) {
  MLIRContext *ctx = b.getContext();
  if (!collapse) {
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
    return b.create<memref::ReinterpretCastOp>(loc, gmTileTy, vi.base,
                                               OpFoldResult(off), sizes, strides);
  }
  SmallVector<unsigned> kept = getNonPinnedDims(vi.tile);
  SmallVector<int64_t> collapsedShape, collapsedStrideStatic;
  SmallVector<OpFoldResult> sizes, strides;
  for (unsigned d : kept) {
    collapsedShape.push_back(vi.tile[d]);
    collapsedStrideStatic.push_back(vi.strideStatic[d]);
    sizes.push_back(b.getIndexAttr(vi.tile[d]));
    if (vi.strideStatic[d] == ShapedType::kDynamic)
      strides.push_back(vi.strideVal[d]);
    else
      strides.push_back(b.getIndexAttr(vi.strideStatic[d]));
  }
  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                       collapsedStrideStatic);
  auto gmTileTy = MemRefType::get(collapsedShape, vi.elementType, layout);
  return b.create<memref::ReinterpretCastOp>(loc, gmTileTy, vi.base,
                                             OpFoldResult(off), sizes, strides);
}

/// Compute the physical tile-origin offset.
static Value computeOffset(OpBuilder &b, Location loc, const ViewData &vi,
                           ValueRange indices) {
  Value off;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal[d]);
    Value logical = b.create<arith::MulIOp>(loc, indices[d], cTrav);
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    off = d == 0 ? phys : b.create<arith::AddIOp>(loc, off, phys);
  }
  return off;
}

/// Return the in-bounds prefix of a memref.
static Value createPrefixSubview(OpBuilder &b, Location loc, Value src,
                                 ValueRange lens) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  for (Value len : lens) {
    offsets.push_back(b.getIndexAttr(0));
    sizes.push_back(OpFoldResult(len));
    strides.push_back(b.getIndexAttr(1));
  }
  return b.create<memref::SubViewOp>(loc, src, offsets, sizes, strides);
}

/// Test for the unbounded-extent sentinel emitted by Pass A.
static bool isSentinelExtent(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue() == std::numeric_limits<int64_t>::max();
  return false;
}

/// Compute the in-bounds tile lengths and physical origin.
static Value createTailGeometry(OpBuilder &b, Location loc, const ViewData &vi,
                                ValueRange indices, SmallVectorImpl<Value> &lens) {
  Value off;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal[d]);
    Value offLog = b.create<arith::MulIOp>(loc, indices[d], cTrav);
    Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    Value nMinus = b.create<arith::SubIOp>(loc, vi.n[d], offLog);
    lens.push_back(b.create<arith::MinSIOp>(loc, nMinus, cTile));
    Value phys = b.create<arith::MulIOp>(loc, offLog, vi.strideVal[d]);
    off = d == 0 ? phys : b.create<arith::AddIOp>(loc, off, phys);
  }
  return off;
}

static Value asIndex(OpBuilder &b, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
}

/// Reinterpret a scalar GM element at the given offset.
static Value createScalarGmElem(OpBuilder &b, Location loc, Value base, Value ik,
                                Type elemTy) {
  auto layout = StridedLayoutAttr::get(b.getContext(),
                                       /*offset=*/ShapedType::kDynamic, {1});
  auto ty = MemRefType::get({1}, elemTy, layout);
  return b.create<memref::ReinterpretCastOp>(
      loc, ty, base, OpFoldResult(ik),
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

/// Compute the contiguous block geometry for one sparse index.
static void createBlockGeometry(OpBuilder &b, Location loc, const ViewData &vi,
                                ValueRange indices, unsigned sparseDim, Value sIdx,
                                Value &off, SmallVectorImpl<OpFoldResult> &sizes,
                                SmallVectorImpl<OpFoldResult> &strides,
                                SmallVectorImpl<OpFoldResult> &subOffs) {
  for (unsigned d = 0; d < vi.rank; ++d) {
    int64_t blk = (d == sparseDim) ? 1 : vi.tile[d];
    Value logical =
        d == sparseDim
            ? sIdx
            : b.create<arith::MulIOp>(
                  loc, indices[d], b.create<arith::ConstantIndexOp>(loc, vi.tile[d]));
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    off = off ? b.create<arith::AddIOp>(loc, off, phys) : phys;
    sizes.push_back(b.getIndexAttr(blk));
    strides.push_back(vi.strideStatic[d] == ShapedType::kDynamic
                          ? OpFoldResult(vi.strideVal[d])
                          : OpFoldResult(b.getIndexAttr(vi.strideStatic[d])));
    subOffs.push_back(b.getIndexAttr(0)); // sparse offset overwritten by caller
  }
}

/// Gather contiguous blocks along one sparse dimension.
static Value createBlockGather(OpBuilder &b, Location loc, const ViewData &vi,
                               Value base, ValueRange indices, unsigned sparseDim,
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
    auto ext = b.create<tensor::ExtractOp>(loc, indices[sparseDim], ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value sIdx = asIndex(b, loc, ext.getResult());

    Value off;
    SmallVector<OpFoldResult> sizes, strides, subOffs;
    createBlockGeometry(b, loc, vi, indices, sparseDim, sIdx, off, sizes, strides,
                        subOffs);
    subOffs[sparseDim] = OpFoldResult(k);
    SmallVector<OpFoldResult> subStr(vi.rank, b.getIndexAttr(1));
    Value gm = b.create<memref::ReinterpretCastOp>(loc, gmBlkTy, base,
                                                   OpFoldResult(off), sizes, strides);
    Value sub = b.create<memref::SubViewOp>(loc, buf, subOffs, sizes, subStr);
    b.create<memref::CopyOp>(loc, gm, sub);
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());
  return b.create<bufferization::ToTensorOp>(loc, resultType, buf,
                                             /*restrict=*/true, /*writable=*/false);
}

/// Scatter contiguous blocks along one sparse dimension.
static void createBlockScatter(OpBuilder &b, Location loc, const ViewData &vi,
                               Value base, Value value, ValueRange indices,
                               unsigned sparseDim) {
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
    auto ext = b.create<tensor::ExtractOp>(loc, indices[sparseDim], ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value sIdx = asIndex(b, loc, ext.getResult());

    Value off;
    SmallVector<OpFoldResult> sizes, strides, subOffs;
    createBlockGeometry(b, loc, vi, indices, sparseDim, sIdx, off, sizes, strides,
                        subOffs);
    subOffs[sparseDim] = OpFoldResult(k);
    SmallVector<OpFoldResult> subStr(vi.rank, b.getIndexAttr(1));
    Value gm = b.create<memref::ReinterpretCastOp>(loc, gmBlkTy, base,
                                                   OpFoldResult(off), sizes, strides);
    Value slice =
        b.create<tensor::ExtractSliceOp>(loc, value, subOffs, sizes, subStr);
    auto mat = b.create<bufferization::MaterializeInDestinationOp>(loc, slice, gm);
    mat->setAttr("writable", b.getUnitAttr());
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());
}

/// Return whether a sparse access contains a regular contiguous block.
static bool canBlockCopy(const ViewData &vi, unsigned sparseDim) {
  for (unsigned d = 0; d < vi.rank; ++d)
    if (d != sparseDim && vi.strideStatic[d] == 1)
      return true;
  return false;
}

/// Compute a scalar gather/scatter offset for regular and sparse dimensions.
/// Scalar accesses intentionally omit the block-DMA marker.
static Value computeElementOffset(OpBuilder &b, Location loc,
                                  const ViewData &vi, ValueRange indices,
                                  ArrayRef<int64_t> sparseDims,
                                  ValueRange coords) {
  Value offset;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value logical;
    if (llvm::is_contained(sparseDims, static_cast<int64_t>(d))) {
      auto ext = b.create<tensor::ExtractOp>(loc, indices[d], coords[d]);
      logical = asIndex(b, loc, ext.getResult());
    } else {
      Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
      Value origin = b.create<arith::MulIOp>(loc, indices[d], cTile);
      logical = b.create<arith::AddIOp>(loc, origin, coords[d]);
    }
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    offset = offset ? b.create<arith::AddIOp>(loc, offset, phys) : phys;
  }
  return offset;
}

/// Widen masks before per-lane extraction to avoid packed-i1 vectorization.
static Value widenMaskToI8(OpBuilder &b, Location loc, Value mask) {
  auto maskTy = cast<RankedTensorType>(mask.getType());
  auto wideTy = RankedTensorType::get(maskTy.getShape(), b.getI8Type());
  return b.create<arith::ExtUIOp>(loc, wideTy, mask);
}

/// Extract a boolean lane from a widened mask.
static Value extractWideMaskElem(OpBuilder &b, Location loc, Value maskWide,
                                 ValueRange coords) {
  Value elem = b.create<tensor::ExtractOp>(loc, maskWide, coords);
  Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(b.getI8Type()));
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, elem, zero);
}

/// Gather elementwise when block DMA cannot represent the sparse access.
static Value createElementwiseGather(OpBuilder &b, Location loc,
                                     const ViewData &vi, Value base,
                                     ValueRange indices,
                                     ArrayRef<int64_t> sparseDims, Value mask,
                                     RankedTensorType resultType) {
  Value init = b.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                         resultType.getElementType());
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value maskWide = mask ? widenMaskToI8(b, loc, mask) : Value();
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
  auto doLoad = [&](OpBuilder &bb) -> Value {
    Value offset = computeElementOffset(bb, loc, vi, indices, sparseDims, coords);
    Value rc = createScalarGmElem(bb, loc, base, offset, vi.elementType);
    return bb.create<memref::LoadOp>(loc, rc, ValueRange{c0});
  };

  Value result;
  if (mask) {
    Value maskElem = extractWideMaskElem(b, loc, maskWide, coords);
    auto ifOp = b.create<scf::IfOp>(
        loc, maskElem,
        [&](OpBuilder &bb, Location loc) {
          Value v = doLoad(bb);
          Value ins = bb.create<tensor::InsertOp>(loc, v, target, coords);
          bb.create<scf::YieldOp>(loc, ins);
        },
        [&](OpBuilder &bb, Location loc) {
          Value zero = bb.create<arith::ConstantOp>(loc, bb.getZeroAttr(vi.elementType));
          Value ins = bb.create<tensor::InsertOp>(loc, zero, target, coords);
          bb.create<scf::YieldOp>(loc, ins);
        });
    result = ifOp.getResult(0);
  } else {
    Value v = doLoad(b);
    result = b.create<tensor::InsertOp>(loc, v, target, coords);
  }
  b.create<scf::YieldOp>(loc, result);
  b.setInsertionPointAfter(loops.front());
  return loops.front().getResult(0);
}

/// Scatter elementwise when block DMA cannot represent the sparse access.
static void createElementwiseScatter(OpBuilder &b, Location loc,
                                     const ViewData &vi, Value base,
                                     Value value, ValueRange indices,
                                     ArrayRef<int64_t> sparseDims, Value mask) {
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  OpBuilder::InsertionGuard guard(b);
  Value maskWide = mask ? widenMaskToI8(b, loc, mask) : Value();
  SmallVector<Value> coords;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, c0, upper, c1);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  auto doStore = [&](OpBuilder &bb) {
    Value offset = computeElementOffset(bb, loc, vi, indices, sparseDims, coords);
    auto elem = bb.create<tensor::ExtractOp>(loc, value, coords);
    Value rc = createScalarGmElem(bb, loc, base, offset, vi.elementType);
    Value empty =
        bb.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{1}, vi.elementType);
    Value ins = bb.create<tensor::InsertOp>(loc, elem.getResult(), empty,
                                            ValueRange{c0});
    auto mat = bb.create<bufferization::MaterializeInDestinationOp>(loc, ins, rc);
    mat->setAttr("writable", bb.getUnitAttr());
  };

  if (mask) {
    Value maskElem = extractWideMaskElem(b, loc, maskWide, coords);
    b.create<scf::IfOp>(loc, maskElem, [&](OpBuilder &bb, Location loc) {
      doStore(bb);
      bb.create<scf::YieldOp>(loc);
    });
  } else {
    doStore(b);
  }
}

/// Use block DMA only for an unmasked, non-contiguous single sparse dimension.
static Value createScalarGather(OpBuilder &b, Location loc, const ViewData &vi,
                                Value base, ValueRange indices,
                                ArrayRef<int64_t> sparseDims, Value mask,
                                RankedTensorType resultType) {
  if (!mask && sparseDims.size() == 1 && canBlockCopy(vi, sparseDims[0]))
    return createBlockGather(b, loc, vi, base, indices, sparseDims[0], resultType);
  return createElementwiseGather(b, loc, vi, base, indices, sparseDims, mask,
                                 resultType);
}

static void createScalarScatter(OpBuilder &b, Location loc, const ViewData &vi,
                                Value base, Value value, ValueRange indices,
                                ArrayRef<int64_t> sparseDims, Value mask) {
  if (!mask && sparseDims.size() == 1 && canBlockCopy(vi, sparseDims[0]))
    createBlockScatter(b, loc, vi, base, value, indices, sparseDims[0]);
  else
    createElementwiseScatter(b, loc, vi, base, value, indices, sparseDims, mask);
}

//===----------------------------------------------------------------------===//
// View load/store lowering
//===----------------------------------------------------------------------===//

static LogicalResult lowerViewLoad(tv::ViewLoadOp load) {
  ViewData vi = parseView(load.getView());
  if (!vi.ok || load.getIndices().size() != vi.rank)
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  auto tensorTy = cast<RankedTensorType>(load.getResult().getType());

  if (vi.isGatherScatter()) {
    for (unsigned d = 0; d < vi.rank; ++d) {
      bool isSparse = llvm::is_contained(vi.sparseDims, static_cast<int64_t>(d));
      if (isSparse != isa<RankedTensorType>(load.getIndices()[d].getType()))
        return failure();
    }

    // Sparse boundaries are represented by the explicit per-lane mask.
    Value result = createScalarGather(b, loc, vi, vi.base, load.getIndices(),
                                      vi.sparseDims, load.getMask(), tensorTy);
    load.getResult().replaceAllUsesWith(result);
    load.erase();
    return success();
  }

  // The backend assigns the staging buffer's memory space from its consumers.
  SmallVector<unsigned> kept = getNonPinnedDims(vi.tile);
  bool collapsed = kept.size() != vi.rank;
  auto localTy = MemRefType::get(vi.tile, vi.elementType);
  SmallVector<int64_t> collapsedTile;
  for (unsigned d : kept)
    collapsedTile.push_back(vi.tile[d]);
  auto collapsedLocalTy = MemRefType::get(collapsedTile, vi.elementType);
  Value buf = b.create<memref::AllocOp>(loc, collapsed ? collapsedLocalTy : localTy);

  bool masked = false;
  for (Value n : vi.n)
    if (!isSentinelExtent(n)) {
      masked = true;
      break;
    }

  if (!masked) {
    Value off = computeOffset(b, loc, vi, load.getIndices());
    Value gm = createGmTile(b, loc, vi, off, /*collapse=*/true);
    b.create<memref::CopyOp>(loc, gm, buf);
  } else {
    SmallVector<Value> lens;
    Value off = createTailGeometry(b, loc, vi, load.getIndices(), lens);
    Value gm = createGmTile(b, loc, vi, off, /*collapse=*/true);
    // Zero-pad the tile before copying its in-bounds prefix.
    Value zero =
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(vi.elementType));
    b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{buf});
    SmallVector<Value> collapsedLens;
    for (unsigned d : kept)
      collapsedLens.push_back(lens[d]);
    Value gmSub = createPrefixSubview(b, loc, gm, collapsedLens);
    Value bufSub = createPrefixSubview(b, loc, buf, collapsedLens);
    b.create<memref::CopyOp>(loc, gmSub, bufSub);
  }

  Value bufForTensor = buf;
  if (collapsed) {
    auto reassoc = buildPinnedDimReassociation(kept, vi.rank);
    bufForTensor = b.create<memref::ExpandShapeOp>(loc, localTy, buf, reassoc);
  }
  Value t = b.create<bufferization::ToTensorOp>(loc, tensorTy, bufForTensor,
                                                /*restrict=*/true,
                                                /*writable=*/false);
  load.getResult().replaceAllUsesWith(t);
  load.erase();
  return success();
}

static LogicalResult lowerViewStore(tv::ViewStoreOp store) {
  ViewData vi = parseView(store.getView());
  if (!vi.ok || store.getIndices().size() != vi.rank)
    return failure();

  OpBuilder b(store);
  Location loc = store.getLoc();

  if (vi.isGatherScatter()) {
    for (unsigned d = 0; d < vi.rank; ++d) {
      bool isSparse = llvm::is_contained(vi.sparseDims, static_cast<int64_t>(d));
      if (isSparse != isa<RankedTensorType>(store.getIndices()[d].getType()))
        return failure();
    }

    // Sparse boundaries are represented by the explicit per-lane mask.
    createScalarScatter(b, loc, vi, vi.base, store.getValue(),
                        store.getIndices(), vi.sparseDims, store.getMask());
    store.erase();
    return success();
  }

  // The backend requires materialize_in_destination for GM stores. Keep the
  // destination rank because only memref.copy is subject to the DMA rank limit.
  Value value = store.getValue();

  bool masked = false;
  for (Value n : vi.n)
    if (!isSentinelExtent(n)) {
      masked = true;
      break;
    }

  if (!masked) {
    Value off = computeOffset(b, loc, vi, store.getIndices());
    Value gm = createGmTile(b, loc, vi, off, /*collapse=*/false);
    auto mat = b.create<bufferization::MaterializeInDestinationOp>(loc, value, gm);
    mat->setAttr("writable", b.getUnitAttr());
  } else {
    SmallVector<Value> lens;
    Value off = createTailGeometry(b, loc, vi, store.getIndices(), lens);
    Value gm = createGmTile(b, loc, vi, off, /*collapse=*/false);
    Value gmSub = createPrefixSubview(b, loc, gm, lens);
    // Store only the in-bounds prefix.
    SmallVector<OpFoldResult> offs, szs, strs;
    for (Value len : lens) {
      offs.push_back(b.getIndexAttr(0));
      szs.push_back(OpFoldResult(len));
      strs.push_back(b.getIndexAttr(1));
    }
    Value slice = b.create<tensor::ExtractSliceOp>(loc, value, offs, szs, strs);
    auto mat =
        b.create<bufferization::MaterializeInDestinationOp>(loc, slice, gmSub);
    mat->setAttr("writable", b.getUnitAttr());
  }

  store.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Discrete pointer accesses
//===----------------------------------------------------------------------===//

static LogicalResult lowerPtrLoad(tv::PtrLoadOp op) {
  // Read the base untyped after function arguments have become memrefs.
  Value base = op->getOperand(0);
  if (!isa<MemRefType>(base.getType()) || op.getIndices().empty())
    return failure();
  Value idx = op.getIndices()[0]; // tensor<Nxindex>
  Value count = op.getCount();    // optional valid-lane count (null if absent)
  auto resTy = cast<RankedTensorType>(op.getResult().getType());
  int64_t n = resTy.getShape()[0];
  if (ShapedType::isDynamic(n))
    return failure();
  Type elemTy = resTy.getElementType();

  OpBuilder b(op);
  Location loc = op.getLoc();
  auto bufTy = MemRefType::get({n}, elemTy);
  Value buf = b.create<memref::AllocOp>(loc, bufTy);
  Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(elemTy));
  b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{buf});

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  // Limit the loop to valid lanes; the initialized pad covers the tail.
  Value ub = count ? count : b.create<arith::ConstantIndexOp>(loc, n);
  auto loop = b.create<scf::ForOp>(loc, c0, ub, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    auto ext = b.create<tensor::ExtractOp>(loc, idx, ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value rc = createScalarGmElem(b, loc, base, ext.getResult(), elemTy);
    Value v = b.create<memref::LoadOp>(loc, rc, ValueRange{c0});
    b.create<memref::StoreOp>(loc, v, buf, ValueRange{k});
  }

  Value t = b.create<bufferization::ToTensorOp>(loc, resTy, buf,
                                                /*restrict=*/true,
                                                /*writable=*/false);
  op.getResult().replaceAllUsesWith(t);
  op.erase();
  return success();
}

static LogicalResult lowerPtrStore(tv::PtrStoreOp op) {
  // Read the base untyped after function arguments have become memrefs.
  Value base = op->getOperand(0);
  if (!isa<MemRefType>(base.getType()) || op.getIndices().empty())
    return failure();
  Value idx = op.getIndices()[0];
  Value count = op.getCount(); // optional valid-lane count (null if absent)
  Value value = op.getValue();
  auto valTy = cast<RankedTensorType>(value.getType());
  int64_t n = valTy.getShape()[0];
  if (ShapedType::isDynamic(n))
    return failure();
  Type elemTy = valTy.getElementType();

  OpBuilder b(op);
  Location loc = op.getLoc();
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  // Do not scatter padded tail lanes.
  Value ub = count ? count : b.create<arith::ConstantIndexOp>(loc, n);
  auto loop = b.create<scf::ForOp>(loc, c0, ub, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    // Use the backend-supported tensor materialization form for GM writes.
    auto extIdx = b.create<tensor::ExtractOp>(loc, idx, ValueRange{k});
    extIdx->setAttr("DiscreteMemAccess", b.getUnitAttr());
    auto extVal = b.create<tensor::ExtractOp>(loc, value, ValueRange{k});
    extVal->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value rc = createScalarGmElem(b, loc, base, extIdx.getResult(), elemTy);
    Value empty = b.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{1}, elemTy);
    Value ins =
        b.create<tensor::InsertOp>(loc, extVal.getResult(), empty, ValueRange{c0});
    auto mat = b.create<bufferization::MaterializeInDestinationOp>(loc, ins, rc);
    mat->setAttr("writable", b.getUnitAttr());
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());

  op.erase();
  return success();
}

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
    : public mlir::triton::impl::TensorViewLoweringBase<TensorViewLoweringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](triton::FuncOp func) { rewriteFuncPtrArgs(func); });

    // Collect before rewriting to avoid mutating during the walk.
    SmallVector<tv::ViewLoadOp> loads;
    SmallVector<tv::ViewStoreOp> stores;
    SmallVector<tv::PtrLoadOp> ptrLoads;
    SmallVector<tv::PtrStoreOp> ptrStores;
    module.walk([&](Operation *op) {
      if (auto l = dyn_cast<tv::ViewLoadOp>(op))
        loads.push_back(l);
      else if (auto s = dyn_cast<tv::ViewStoreOp>(op))
        stores.push_back(s);
      else if (auto pl = dyn_cast<tv::PtrLoadOp>(op))
        ptrLoads.push_back(pl);
      else if (auto ps = dyn_cast<tv::PtrStoreOp>(op))
        ptrStores.push_back(ps);
    });
    for (tv::ViewLoadOp l : loads)
      if (failed(lowerViewLoad(l)))
        l.emitError("TensorViewLowering: unsupported view_load");
    for (tv::ViewStoreOp s : stores)
      if (failed(lowerViewStore(s)))
        s.emitError("TensorViewLowering: unsupported view_store");
    for (tv::PtrLoadOp pl : ptrLoads)
      if (failed(lowerPtrLoad(pl)))
        pl.emitError("TensorViewLowering: unsupported ptr_load");
    for (tv::PtrStoreOp ps : ptrStores)
      if (failed(lowerPtrStore(ps)))
        ps.emitError("TensorViewLowering: unsupported ptr_store");

    // Erase dead view construction operations.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      module.walk([&](Operation *op) {
        if ((isa<tv::MakePartitionViewOp>(op) ||
             isa<tv::MakeStridedViewOp>(op) ||
             isa<tv::MakeGatherScatterViewOp>(op) ||
             isa<tv::MakeTensorViewOp>(op)) &&
            op->use_empty())
          dead.push_back(op);
      });
      for (Operation *op : dead) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTensorViewLoweringPass() {
  return std::make_unique<TensorViewLoweringPass>();
}
