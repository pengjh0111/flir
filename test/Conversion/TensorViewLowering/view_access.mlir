// RUN: triton-shared-opt --tensor-view-lowering %s | FileCheck %s

module {
  tt.func public @view_load(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : <f32> -> <?xf32, strides=[?]>
    %view = tv.make_partition_view %base_view : <?xf32, strides=[?]> -> <?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>
    %result = tv.view_load %view[%index] : <?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>, index -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  tt.func public @view_store(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index, %value: tensor<4xf32>) {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : <f32> -> <?xf32, strides=[?]>
    %view = tv.make_strided_view %base_view : <?xf32, strides=[?]> -> <?xf32, strides=[?], #tv.strided_view<tile = [4], dim_map = [0], traversal_strides = [8], padding_value = inf>>
    tv.view_store %view[%index], %value : <?xf32, strides=[?], #tv.strided_view<tile = [4], dim_map = [0], traversal_strides = [8], padding_value = inf>>, tensor<4xf32>, index
    tt.return
  }

  tt.func public @gather_padding(%base: !tv.ptr<f32>, %size: index, %stride: index, %indices: tensor<4xi32>) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : <f32> -> <?xf32, strides=[?]>
    %view = tv.make_gather_scatter_view %base_view : <?xf32, strides=[?]> -> <?xf32, strides=[?], #tv.gather_scatter_view<tile = [4], sparse_dim = [0], padding_value = nan>>
    %result = tv.view_load %view[%indices] : <?xf32, strides=[?], #tv.gather_scatter_view<tile = [4], sparse_dim = [0], padding_value = nan>>, tensor<4xi32> -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  tt.func public @negative_infinity_padding(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : <f32> -> <?xf32, strides=[?]>
    %view = tv.make_partition_view %base_view : <?xf32, strides=[?]> -> <?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = -inf>>
    %result = tv.view_load %view[%index] : <?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = -inf>>, index -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  tt.func public @integer_zero_padding(%base: !tv.ptr<i32>, %size: index, %stride: index, %index: index) -> tensor<4xi32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : <i32> -> <?xi32, strides=[?]>
    %view = tv.make_partition_view %base_view : <?xi32, strides=[?]> -> <?xi32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>
    %result = tv.view_load %view[%index] : <?xi32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>, index -> tensor<4xi32>
    tt.return %result : tensor<4xi32>
  }
}

// CHECK-LABEL: tt.func public @view_load(
// CHECK-SAME: memref<?xf32>
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK-NOT: tv.
// CHECK: tt.return

// CHECK-LABEL: tt.func public @view_store(
// CHECK-SAME: memref<?xf32>
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: memref.reinterpret_cast
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: tensor.empty() : tensor<1xf32>
// CHECK: memref.store
// CHECK-NOT: tv.
// CHECK: tt.return

// CHECK-LABEL: tt.func public @gather_padding(
// CHECK-SAME: memref<?xf32>
// CHECK: scf.for
// CHECK: scf.if
// CHECK: arith.constant {{.*}} : f32
// CHECK: memref.load
// CHECK-NOT: tv.
// CHECK: tt.return

// CHECK-LABEL: tt.func public @negative_infinity_padding(
// CHECK: scf.if
// CHECK: arith.constant {{.*}} : f32
// CHECK-NOT: tv.
// CHECK: tt.return

// CHECK-LABEL: tt.func public @integer_zero_padding(
// CHECK: scf.if
// CHECK: arith.constant 0 : i32
// CHECK-NOT: tv.
// CHECK: tt.return
