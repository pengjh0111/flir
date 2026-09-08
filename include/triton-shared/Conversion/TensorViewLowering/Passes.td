//===-- Passes.td - TensorViewLowering pass definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_LOWERING_CONVERSION_PASSES
#define TENSORVIEW_LOWERING_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TensorViewLowering : Pass<"tensor-view-lowering", "mlir::ModuleOp"> {
  let summary = "Lower TensorView access ops to generic IR";
  let description = [{
    GENERIC lowering variant: lowers the TensorView access ops to *community*
    memref/bufferization/scf/tensor ops and scalar Triton memory accesses (no
    HIVM dialect), mirroring the native tt->linalg access form so the backend
    does placement + DMA lowering:
      - `!tv.ptr<T>` function inputs -> plain `memref<?xT>`
      - in-bounds `tv.view_load` -> reinterpret_cast + alloc + memref.copy +
                                    to_tensor
      - in-bounds `tv.view_store` -> reinterpret_cast +
                                     materialize_in_destination
      - boundary view accesses -> scalar bounds checks with view padding
      - gather/scatter view -> scalar scf.for over the tile (reinterpret sizes:[1]
                               + memref.load / materialize_in_destination)
      - ptr_load/ptr_store -> scalar scf.for with scalar tt.load/tt.store
  }];
  let constructor = "triton::createTensorViewLoweringPass()";
  let dependentDialects = [
    "mlir::triton::tv::TensorViewDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::triton::TritonDialect"];
}

#endif // TENSORVIEW_LOWERING_CONVERSION_PASSES
