//===- Passes.h - TritonToTensorView passes ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_TRITON_TO_TENSORVIEW_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TRITON_TO_TENSORVIEW_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Converts supported discrete pointer accesses to TensorView ptr operations.
std::unique_ptr<OperationPass<ModuleOp>> createTritonToTensorViewPass();

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToTensorView/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_TENSORVIEW_CONVERSION_PASSES_H
