// RUN: triton-shared-opt %s --verify-diagnostics
#view = #tv.gather_scatter_view<tile = [4], sparse_dim = [0], padding_value = zero>
func.func @bad_coordinates(%base: !tv.ptr<f32>, %size: index, %stride: index, %coordinates: memref<3xi32>, %buffer: memref<4xf32>) {
  %v = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
  %g = tv.make_gather_scatter_view %v : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #view>
  // expected-error @+1 {{sparse coordinates must be rank-1 index/i32/i64 buffers matching the tile}}
  tv.view_load_into %g[%coordinates], %buffer : !tv.tensor_view<?xf32, strides=[?], #view>, memref<4xf32>, memref<3xi32>
  return
}
