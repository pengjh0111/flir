//===- TensorViewBuilder.h - Direct TensorView construction -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_CONVERSION_TRITON_TO_TENSOR_VIEW_BUILDER_H
#define TRITON_SHARED_CONVERSION_TRITON_TO_TENSOR_VIEW_BUILDER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace triton {
namespace tv {

// Build an unencoded view from a scalar pointer, shape, and element strides.
Value createTensorViewBase(OpBuilder &b, Location loc, Value basePtr,
                           ValueRange shape, ValueRange strides);

// Access an encoded tile. Sparse dimensions use tensor indices and an optional
// per-lane mask; regular dimensions derive boundaries from the view shape.
Value tensorViewLoad(OpBuilder &b, Location loc, Value baseView,
                     ArrayRef<int64_t> tile, ArrayRef<int64_t> traversal,
                     ArrayRef<int64_t> sparseDims, ValueRange index,
                     Type resultTy, Value mask);
void tensorViewStore(OpBuilder &b, Location loc, Value baseView,
                     ArrayRef<int64_t> tile, ArrayRef<int64_t> traversal,
                     ArrayRef<int64_t> sparseDims, ValueRange index,
                     Value value, Value mask);

} // namespace tv
} // namespace triton
} // namespace mlir

#endif // TRITON_SHARED_CONVERSION_TRITON_TO_TENSOR_VIEW_BUILDER_H
