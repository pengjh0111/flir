// RUN: triton-shared-opt %s | FileCheck %s
// The quoted negative-infinity spelling is also readable by the LLVM 19 backend.
#negative = #tv.partition_view<tile = [4], dim_map = [0], padding_value = "-inf">
func.func @negative_padding(%view: !tv.tensor_view<?xf32, strides=[1], #negative>) {
  return
}
// CHECK: padding_value = "-inf"
