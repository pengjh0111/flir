//===- TritonToTensorView.cpp - tt -> tv conversion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts supported rank-1 discrete pointer accesses to TensorView ptr
// operations. Unmatched accesses remain on the native lowering path.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToTensorView/Passes.h"

#include "triton-shared/Conversion/TritonToTensorView/Matcher.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOTENSORVIEW
#include "triton-shared/Conversion/TritonToTensorView/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace tv = mlir::triton::tv;

namespace {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Return the pointee type of a Triton or TensorView pointer.
static Type getPtrPointeeType(Type t) {
  if (auto p = dyn_cast<tv::PtrType>(t))
    return p.getPointeeType();
  if (auto p = dyn_cast<triton::PointerType>(t))
    return p.getPointeeType();
  return Type();
}

/// True for a regular, arange-based tile offset (`make_range`, or
/// `origin + make_range`) -- a dense access that belongs on the native path,
/// not the discrete pointer fallback.
static bool isRegularAffineIndex(Value idxTensor) {
  if (idxTensor.getDefiningOp<triton::MakeRangeOp>())
    return true;
  auto addi = idxTensor.getDefiningOp<arith::AddIOp>();
  if (!addi)
    return false;
  bool hasRange = false, hasOrigin = false;
  for (Value op : {addi.getLhs(), addi.getRhs()}) {
    hasRange |= bool(op.getDefiningOp<triton::MakeRangeOp>());
    hasOrigin |= bool(op.getDefiningOp<triton::SplatOp>());
  }
  return hasRange && hasOrigin;
}

/// Match the rank-1 discrete pointer fallback, folding hoisted scalar offsets
/// into the index tensor.
static bool matchPtrAccess(OpBuilder &b, Location loc, Value ptrTensor,
                           Value &basePtr, Value &idxTensor) {
  auto addptr = ptrTensor.getDefiningOp<triton::AddPtrOp>();
  if (!addptr)
    return false;
  auto splat = addptr.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splat)
    return false;
  // Discrete offsets must be ranked tensors, and not a regular tile access.
  if (!isa<RankedTensorType>(addptr.getOffset().getType()))
    return false;
  if (isRegularAffineIndex(addptr.getOffset()))
    return false;

  Value scalarPtr = splat.getSrc();
  Value scalarOffset;
  if (auto scalarAp = scalarPtr.getDefiningOp<triton::AddPtrOp>()) {
    scalarOffset = scalarAp.getOffset();
    scalarPtr = scalarAp.getPtr();
  }
  if (!getPtrPointeeType(scalarPtr.getType()))
    return false;

  basePtr = scalarPtr;
  idxTensor = addptr.getOffset();
  if (scalarOffset) {
    auto idxTy = cast<RankedTensorType>(idxTensor.getType());
    Value splatOff = b.create<triton::SplatOp>(
        loc, RankedTensorType::get(idxTy.getShape(), scalarOffset.getType()),
        scalarOffset);
    idxTensor = b.create<arith::AddIOp>(loc, idxTensor, splatOff);
  }
  return true;
}

/// Cast an integer index tensor to `tensor<...xindex>` (tv index tensor type).
static Value toIndexTensor(OpBuilder &b, Location loc, Value idxTensor) {
  auto ty = dyn_cast<RankedTensorType>(idxTensor.getType());
  if (!ty)
    return idxTensor;
  if (ty.getElementType().isIndex())
    return idxTensor;
  return b.create<arith::IndexCastOp>(
      loc, RankedTensorType::get(ty.getShape(), b.getIndexType()), idxTensor);
}

/// Recover the valid-lane count from a contiguous-prefix mask.
static Value computePtrCount(OpBuilder &b, Location loc, Value mask,
                             int64_t blk) {
  if (!mask)
    return Value();
  auto cmp = mask.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return Value();
  Value bound, offExpr;
  for (Value op : {cmp.getLhs(), cmp.getRhs()}) {
    if (auto s = op.getDefiningOp<triton::SplatOp>())
      bound = s.getSrc();
    else
      offExpr = op;
  }
  if (!bound || !offExpr)
    return Value();
  Value origin;
  if (auto addi = offExpr.getDefiningOp<arith::AddIOp>())
    for (Value op : {addi.getLhs(), addi.getRhs()})
      if (auto s = op.getDefiningOp<triton::SplatOp>())
        origin = s.getSrc();

  Value boundI = b.create<arith::IndexCastOp>(loc, b.getIndexType(), bound);
  Value diff = boundI;
  if (origin) {
    Value originI = b.create<arith::IndexCastOp>(loc, b.getIndexType(), origin);
    diff = b.create<arith::SubIOp>(loc, boundI, originI);
  }
  Value cBlk = b.create<arith::ConstantIndexOp>(loc, blk);
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value len = b.create<arith::MinSIOp>(loc, diff, cBlk);
  return b.create<arith::MaxSIOp>(loc, len, c0);
}

/// Recursively erase dead pointer-chain operations after inline matching.
static void eraseIfDeadPtrChain(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return;
  if (!isa<triton::AddPtrOp, triton::SplatOp, triton::BroadcastOp,
           triton::ExpandDimsOp>(op))
    return;
  if (!op->use_empty())
    return;
  SmallVector<Value> operands(op->getOperands().begin(),
                              op->getOperands().end());
  op->erase();
  for (Value operand : operands)
    eraseIfDeadPtrChain(operand);
}

} // namespace

//===----------------------------------------------------------------------===//
// Public matcher entry points shared by the inline hook and Pass A.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace tv {

Value ensureTvPtr(Value v) {
  if (isa<tv::PtrType>(v.getType()))
    return v;
  auto tt = dyn_cast<triton::PointerType>(v.getType());
  if (!tt)
    return v;
  auto tvTy = tv::PtrType::get(tt.getPointeeType());
  // Preserve values that still feed an operation requiring !tt.ptr.
  bool feedsLiveMakeTensorPtr = llvm::any_of(v.getUsers(), [&](Operation *u) {
    auto mkPtr = dyn_cast<triton::MakeTensorPtrOp>(u);
    return mkPtr && mkPtr.getBase() == v;
  });
  if (auto ba = dyn_cast<BlockArgument>(v); ba && !feedsLiveMakeTensorPtr) {
    // Block arguments can be retyped with their function signature.
    v.setType(tvTy);
    Block *blk = ba.getOwner();
    if (auto fn = dyn_cast<triton::FuncOp>(blk->getParentOp())) {
      SmallVector<Type> inputs(blk->getArgumentTypes().begin(),
                               blk->getArgumentTypes().end());
      fn.setFunctionType(FunctionType::get(fn.getContext(), inputs,
                                           fn.getFunctionType().getResults()));
    }
    return v;
  }
  // Interior values and shared block arguments require a bridge cast.
  OpBuilder b(v.getContext());
  b.setInsertionPointAfterValue(v);
  return b.create<UnrealizedConversionCastOp>(v.getLoc(), tvTy, v).getResult(0);
}

Value tryEmitTvLoad(OpBuilder &b, Location loc, Value ptr, Value mask,
                    Type resultTy) {
  // Discrete rank-1 pointer access -> ptr_load.
  Value base, idxTensor;
  auto resTy = dyn_cast<RankedTensorType>(resultTy);
  if (resTy && resTy.getRank() == 1 &&
      matchPtrAccess(b, loc, ptr, base, idxTensor)) {
    base = ensureTvPtr(base);
    Value idx = toIndexTensor(b, loc, idxTensor);
    Value count = computePtrCount(b, loc, mask, resTy.getShape()[0]);
    auto pad = tv::PadKindAttr::get(b.getContext(), tv::PadKind::Zero);
    auto ptrLoad = b.create<tv::PtrLoadOp>(loc, resTy, base, ValueRange{idx},
                                           /*count=*/count, pad);
    eraseIfDeadPtrChain(ptr);
    return ptrLoad.getResult();
  }
  return Value();
}

bool tryEmitTvStore(OpBuilder &b, Location loc, Value ptr, Value value,
                    Value mask) {
  // Discrete rank-1 pointer access -> ptr_store.
  Value base, idxTensor;
  auto valTy = dyn_cast<RankedTensorType>(value.getType());
  if (valTy && valTy.getRank() == 1 &&
      matchPtrAccess(b, loc, ptr, base, idxTensor)) {
    base = ensureTvPtr(base);
    Value idx = toIndexTensor(b, loc, idxTensor);
    Value count = computePtrCount(b, loc, mask, valTy.getShape()[0]);
    auto pad = tv::PadKindAttr::get(b.getContext(), tv::PadKind::Zero);
    b.create<tv::PtrStoreOp>(loc, base, value, ValueRange{idx},
                             /*count=*/count, pad);
    eraseIfDeadPtrChain(ptr);
    return true;
  }
  return false;
}

} // namespace tv
} // namespace triton
} // namespace mlir

namespace {

/// Detect synthetic loads used only by bufferization bridge casts.
static bool isBufferizationCastBridge(triton::LoadOp load) {
  if (!load.getResult().hasOneUse())
    return false;
  Operation *user = *load.getResult().getUsers().begin();
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  return isa<bufferization::ToMemrefOp>(user);
#else
  return isa<bufferization::ToBufferOp>(user);
#endif
}

/// Detect synthetic stores fed by bufferization bridge casts.
static bool isBufferizationCastBridge(triton::StoreOp store) {
  return isa_and_nonnull<bufferization::ToTensorOp>(
      store.getValue().getDefiningOp());
}

static LogicalResult rewriteLoad(triton::LoadOp load) {
  if (isBufferizationCastBridge(load))
    return failure();
  OpBuilder b(load);
  Value result = tv::tryEmitTvLoad(b, load.getLoc(), load.getPtr(),
                                   load.getMask(), load.getResult().getType());
  if (!result)
    return failure();
  load.getResult().replaceAllUsesWith(result);
  load.erase();
  return success();
}

static LogicalResult rewriteStore(triton::StoreOp store) {
  if (isBufferizationCastBridge(store))
    return failure();
  OpBuilder b(store);
  if (tv::tryEmitTvStore(b, store.getLoc(), store.getPtr(), store.getValue(),
                         store.getMask())) {
    store.erase();
    return success();
  }
  return failure();
}

/// Erase dead pointer-chain operations to a fixed point.
static void eraseDeadPtrOps(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if ((isa<triton::AddPtrOp>(op) || isa<triton::SplatOp>(op) ||
           isa<triton::BroadcastOp>(op) || isa<triton::ExpandDimsOp>(op) ||
           isa<triton::AdvanceOp>(op) || isa<triton::MakeTensorPtrOp>(op)) &&
          op->use_empty())
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TritonToTensorViewPass
    : public mlir::triton::impl::TritonToTensorViewBase<
          TritonToTensorViewPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect before rewriting to avoid mutating during the walk.
    SmallVector<triton::LoadOp> loads;
    SmallVector<triton::StoreOp> stores;
    module.walk([&](Operation *op) {
      if (auto l = dyn_cast<triton::LoadOp>(op))
        loads.push_back(l);
      else if (auto s = dyn_cast<triton::StoreOp>(op))
        stores.push_back(s);
    });

    // Only supported discrete pointer accesses are rewritten.
    for (triton::LoadOp l : loads)
      (void)rewriteLoad(l);
    for (triton::StoreOp s : stores)
      (void)rewriteStore(s);

    // Clean up the now-dead pointer-chain ops.
    eraseDeadPtrOps(module);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToTensorViewPass() {
  return std::make_unique<TritonToTensorViewPass>();
}
