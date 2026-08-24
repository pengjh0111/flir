//===-- Passes.td - TritonToTensorView pass definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_TENSORVIEW_CONVERSION_PASSES
#define TRITON_TO_TENSORVIEW_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToTensorView : Pass<"triton-to-tensor-view", "mlir::ModuleOp"> {
  let summary = "Convert discrete Triton pointer accesses to TensorView ptr ops";
  let description = [{
    Lowers supported rank-1 pointer-tensor `tt.load`/`tt.store` accesses to
    `tv.ptr_load`/`tv.ptr_store`. Other accesses are left unchanged.
  }];
  let constructor = "triton::createTritonToTensorViewPass()";
  let dependentDialects = [
    "mlir::triton::tv::TensorViewDialect",
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect"];
}

#endif // TRITON_TO_TENSORVIEW_CONVERSION_PASSES
