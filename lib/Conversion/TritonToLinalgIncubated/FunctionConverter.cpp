/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "incubated/Conversion/TritonToLinalgIncubated/FunctionConverter.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace FunctionConverter {
using namespace mlir;
using namespace triton;

LogicalResult GetProgramIDConverter::matchAndRewrite(
    triton::GetProgramIdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto axis = (uint32_t)op.getAxis();
  assert(axis < GetProgramIDConverter::LAUNCH_GRID_RANK &&
         "Invalid axis for GetProgramIdOp");
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto numArgs = func.getNumArguments();
  auto id = func.getArgument(numArgs - GetProgramIDConverter::LAUNCH_GRID_RANK +
                             axis);
  rewriter.replaceOp(op, id);
  return success();
}

LogicalResult GetNumProgramsConverter::matchAndRewrite(
    triton::GetNumProgramsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto axis = (uint32_t)op.getAxis();
  assert(axis < GetNumProgramsConverter::LAUNCH_GRID_RANK &&
         "Invalid axis for GetNumProgramsOp");
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto numArgs = func.getNumArguments();
  auto id = func.getArgument(
      numArgs - GetNumProgramsConverter::LAUNCH_GRID_RANK * 2 + axis);
  rewriter.replaceOp(op, id);
  return success();
}

// Bridge values to the types required by converted function signatures.
static Value bridgeToType(OpBuilder &b, Location loc, Value value,
                          Type targetTy) {
  if (value.getType() == targetTy)
    return value;
  if (isa<TensorType>(value.getType()) && isa<MemRefType>(targetTy))
    return b.create<bufferization::ToBufferOp>(loc, targetTy, value);
  if (isa<MemRefType>(value.getType()) && isa<TensorType>(targetTy))
    return b.create<bufferization::ToTensorOp>(loc, targetTy, value,
                                               /*restrict=*/true,
                                               /*writable=*/false);
  return b.create<mlir::UnrealizedConversionCastOp>(loc, targetTy, value)
      .getResult(0);
}

LogicalResult CallOpConverter::matchAndRewrite(
    triton::CallOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto callee = SymbolTable::lookupNearestSymbolFrom<triton::FuncOp>(
      op, op.getCalleeAttr());
  if (!callee) {
    op->emitError("CallOpConverter: callee symbol not found");
    return failure();
  }
  auto calleeTy = callee.getFunctionType();

  // Convert from the callee's declared signature to avoid rewrite-order
  // dependence between the function and its call sites.
  SmallVector<Type> targetOperandTypes, targetResultTypes;
  if (failed(getTypeConverter()->convertTypes(calleeTy.getInputs(),
                                              targetOperandTypes))) {
    op->emitError("CallOpConverter: convertTypes failed for callee ")
        << callee.getSymName() << "'s " << calleeTy.getNumInputs()
        << " parameter type(s)";
    return failure();
  }
  if (failed(getTypeConverter()->convertTypes(calleeTy.getResults(),
                                              targetResultTypes))) {
    op->emitError("CallOpConverter: convertTypes failed for callee ")
        << callee.getSymName() << "'s " << calleeTy.getNumResults()
        << " result type(s)";
    return failure();
  }
  if (targetOperandTypes.size() != adaptor.getOperands().size()) {
    op->emitError("CallOpConverter: callee ")
        << callee.getSymName() << " declares " << calleeTy.getNumInputs()
        << " parameters (-> " << targetOperandTypes.size()
        << " after conversion), but this tt.call has "
        << adaptor.getOperands().size() << " (adaptor) / "
        << op->getNumOperands() << " (original) operands";
    return failure();
  }
  if (targetResultTypes.size() != op.getNumResults()) {
    op->emitError("CallOpConverter: callee ")
        << callee.getSymName() << " declares " << calleeTy.getNumResults()
        << " results (-> " << targetResultTypes.size()
        << " after conversion), but this tt.call has " << op.getNumResults()
        << " results";
    return failure();
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(targetOperandTypes.size());
  for (auto [operand, targetTy] :
       llvm::zip(adaptor.getOperands(), targetOperandTypes))
    newOperands.push_back(
        bridgeToType(rewriter, op.getLoc(), operand, targetTy));

  auto newCall = rewriter.create<triton::CallOp>(
      op.getLoc(), op.getCalleeAttr(), targetResultTypes, newOperands);

  // Bridge results for consumers that have not yet been converted.
  SmallVector<Value> replacements;
  replacements.reserve(op.getNumResults());
  for (auto [oldResult, newResult] :
       llvm::zip(op.getResults(), newCall.getResults()))
    replacements.push_back(bridgeToType(rewriter, op.getLoc(), newResult,
                                        oldResult.getType()));
  rewriter.replaceOp(op, replacements);
  return success();
}

LogicalResult ReturnOpConverter::matchAndRewrite(
    triton::ReturnOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (adaptor.getSrcs().empty())
    return failure(); // void return: nothing for this pattern to do.

  auto func = op->getParentOfType<triton::FuncOp>();
  SmallVector<Type> targetResultTypes;
  if (failed(getTypeConverter()->convertTypes(
          func.getFunctionType().getResults(), targetResultTypes))) {
    op->emitError("ReturnOpConverter: convertTypes failed for ")
        << func.getSymName();
    return failure();
  }
  if (targetResultTypes.size() != adaptor.getSrcs().size()) {
    op->emitError("ReturnOpConverter: arity mismatch, func ")
        << func.getSymName() << " declares "
        << func.getFunctionType().getResults().size()
        << " results (-> " << targetResultTypes.size()
        << " after conversion), but this tt.return has "
        << adaptor.getSrcs().size() << " (adaptor) / " << op.getSrcs().size()
        << " (original) operands";
    return failure();
  }

  // Always rebuild: adaptor operands may differ from the raw operands used by
  // the dynamic-legality check.
  SmallVector<Value> newOperands;
  newOperands.reserve(targetResultTypes.size());
  for (auto [operand, targetTy] :
       llvm::zip(adaptor.getSrcs(), targetResultTypes)) {
    if (operand.getType() == targetTy) {
      newOperands.push_back(operand);
    } else if (isa<TensorType>(operand.getType()) &&
              isa<MemRefType>(targetTy)) {
      newOperands.push_back(
          rewriter.create<bufferization::ToBufferOp>(op.getLoc(), targetTy,
                                                      operand));
    } else {
      newOperands.push_back(
          rewriter
              .create<mlir::UnrealizedConversionCastOp>(op.getLoc(), targetTy,
                                                        operand)
              .getResult(0));
    }
  }
  rewriter.replaceOpWithNewOp<triton::ReturnOp>(op, newOperands);
  return success();
}
} // namespace FunctionConverter
