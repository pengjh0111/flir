//===- CommonIRDialect.h - MLIR CommonIR dialect -------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CommonIR dialect in MLIR, containing operations that
// explicitly encode memory hierarchy, engine binding, data layout, and
// synchronization for deterministic 1:1 lowering to HIVM.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TILE_IR_DIALECT_H
#define TRITON_DIALECT_TILE_IR_DIALECT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h.inc"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIREnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-ext/Dialect/CommonIR/IR/CommonIROpsAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir-ext/Dialect/CommonIR/IR/CommonIROps.h.inc"

namespace mlir::triton::tile {} // namespace mlir::triton::tile

#endif // TRITON_DIALECT_TILE_IR_DIALECT_H