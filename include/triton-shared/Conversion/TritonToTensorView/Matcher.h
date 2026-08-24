//===- Matcher.h - tt access pattern -> tv, reusable entry points --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structured TensorView accesses are constructed explicitly by the frontend.
// The matcher only handles supported rank-1 discrete pointer accesses.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_CONVERSION_TRITON_TO_TENSOR_VIEW_MATCHER_H
#define TRITON_SHARED_CONVERSION_TRITON_TO_TENSOR_VIEW_MATCHER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace triton {
namespace tv {

// Retype a matched pointer and synchronize its function signature if needed.
Value ensureTvPtr(Value v);

// Emit tv.ptr_load for a supported discrete pointer access, or return null.
Value tryEmitTvLoad(OpBuilder &b, Location loc, Value ptr, Value mask,
                    Type resultTy);

// Emit tv.ptr_store for a supported discrete pointer access, or return false.
bool tryEmitTvStore(OpBuilder &b, Location loc, Value ptr, Value value,
                    Value mask);

} // namespace tv
} // namespace triton
} // namespace mlir

#endif // TRITON_SHARED_CONVERSION_TRITON_TO_TENSOR_VIEW_MATCHER_H
