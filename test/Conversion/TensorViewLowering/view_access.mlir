// RUN: triton-shared-opt --tensor-view-lowering %s | FileCheck %s

module {
  tt.func public @view_load(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
    %view = tv.make_partition_view %base_view : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>
    %result = tv.view_load %view[%index] : !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>, index -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @view_load(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDEX:%.*]]: index) -> tensor<4xf32> {
  // CHECK-DAG: [[PADDING:%.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: [[ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[ORIGIN]], [[C0]] : index
  // CHECK: [[END:%.*]] = arith.addi [[ORIGIN]], [[C4]] : index
  // CHECK: [[WITHIN_EXTENT:%.*]] = arith.cmpi sle, [[END]], [[SIZE]] : index
  // CHECK: [[IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[WITHIN_EXTENT]] : i1
  // CHECK: [[RESULT:%.*]] = scf.if [[IN_BOUNDS]] -> (tensor<4xf32>) {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<4xf32>
  // CHECK: [[TILE_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[TILE_OFFSET:%.*]] = arith.muli [[TILE_ORIGIN]], [[STRIDE]] : index
  // CHECK: [[GM_TILE:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[TILE_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  // CHECK: memref.copy [[GM_TILE]], [[BUFFER]] : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32>
  // CHECK: [[TENSOR:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<4xf32> to tensor<4xf32>
  // CHECK: scf.yield [[TENSOR]] : tensor<4xf32>
  // CHECK: } else {
  // CHECK: [[BOUNDARY_BUFFER:%.*]] = memref.alloc() : memref<4xf32>
  // CHECK: linalg.fill ins([[PADDING]] : f32) outs([[BOUNDARY_BUFFER]] : memref<4xf32>)
  // CHECK: [[BOUNDARY_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[LOWER_CLAMPED:%.*]] = arith.maxsi [[BOUNDARY_ORIGIN]], [[C0]] : index
  // CHECK: [[BEGIN:%.*]] = arith.minsi [[LOWER_CLAMPED]], [[SIZE]] : index
  // CHECK: [[CLIPPED_END:%.*]] = arith.minsi {{%.*}}, [[SIZE]] : index
  // CHECK: [[VALID_SIZE:%.*]] = arith.maxsi {{%.*}}, [[C0]] : index
  // CHECK: [[NON_EMPTY:%.*]] = arith.cmpi sgt, [[VALID_SIZE]], [[C0]] : index
  // CHECK: [[LOCAL_OFFSET:%.*]] = arith.minsi {{%.*}}, [[C4]] : index
  // CHECK: [[BOUNDARY_OFFSET:%.*]] = arith.muli [[BEGIN]], [[STRIDE]] : index
  // CHECK: [[TRANSFER_SIZE:%.*]] = arith.select [[NON_EMPTY]], [[VALID_SIZE]], [[C0]] : index
  // CHECK: [[BOUNDARY_GM:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[BOUNDARY_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  // CHECK: [[BOUNDARY_LOCAL:%.*]] = memref.reinterpret_cast [[BOUNDARY_BUFFER]] to offset: {{.}}[[C0]]{{.}}, sizes: [4], strides: [1] : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: [[GM_SUBVIEW:%.*]] = memref.subview [[BOUNDARY_GM]][0] {{.}}[[TRANSFER_SIZE]]{{.}} [1] : memref<4xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
  // CHECK: [[LOCAL_SUBVIEW:%.*]] = memref.subview [[BOUNDARY_LOCAL]]{{.}}[[LOCAL_OFFSET]]{{.}} {{.}}[[TRANSFER_SIZE]]{{.}} [1] : memref<4xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK: memref.copy [[GM_SUBVIEW]], [[LOCAL_SUBVIEW]] : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK: [[PADDED:%.*]] = bufferization.to_tensor [[BOUNDARY_BUFFER]] restrict : memref<4xf32> to tensor<4xf32>
  // CHECK-NOT: memref.load
  // CHECK: scf.yield [[PADDED]] : tensor<4xf32>
  // CHECK: tt.return [[RESULT]] : tensor<4xf32>

  tt.func public @view_store(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index, %value: tensor<4xf32>) {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
    %view = tv.make_strided_view %base_view : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.strided_view<tile = [4], dim_map = [0], traversal_strides = [8], padding_value = inf>>
    tv.view_store %view[%index], %value : !tv.tensor_view<?xf32, strides=[?], #tv.strided_view<tile = [4], dim_map = [0], traversal_strides = [8], padding_value = inf>>, tensor<4xf32>, index
    tt.return
  }

  // CHECK-LABEL: tt.func public @view_store(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDEX:%.*]]: index, [[VALUE:%.*]]: tensor<4xf32>) {
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C8:%.*]] = arith.constant 8 : index
  // CHECK: [[ORIGIN:%.*]] = arith.muli [[INDEX]], [[C8]] : index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[ORIGIN]], [[C0]] : index
  // CHECK: [[END:%.*]] = arith.addi [[ORIGIN]], [[C4]] : index
  // CHECK: [[WITHIN_EXTENT:%.*]] = arith.cmpi sle, [[END]], [[SIZE]] : index
  // CHECK: [[IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[WITHIN_EXTENT]] : i1
  // CHECK: scf.if [[IN_BOUNDS]] {
  // CHECK: [[TILE_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C8]] : index
  // CHECK: [[TILE_OFFSET:%.*]] = arith.muli [[TILE_ORIGIN]], [[STRIDE]] : index
  // CHECK: [[GM_TILE:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[TILE_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  // CHECK: bufferization.materialize_in_destination [[VALUE]] in writable [[GM_TILE]] : (tensor<4xf32>, memref<4xf32, strided<[?], offset: ?>>) -> ()
  // CHECK: } else {
  // CHECK: [[VALUE_BUFFER:%.*]] = bufferization.to_{{memref|buffer}} [[VALUE]] : tensor<4xf32> to memref<4xf32>
  // CHECK: [[BOUNDARY_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C8]] : index
  // CHECK: [[LOWER_CLAMPED:%.*]] = arith.maxsi [[BOUNDARY_ORIGIN]], [[C0]] : index
  // CHECK: [[BEGIN:%.*]] = arith.minsi [[LOWER_CLAMPED]], [[SIZE]] : index
  // CHECK: [[VALID_SIZE:%.*]] = arith.maxsi {{%.*}}, [[C0]] : index
  // CHECK: [[GM_OFFSET:%.*]] = arith.muli [[BEGIN]], [[STRIDE]] : index
  // CHECK: [[TRANSFER_SIZE:%.*]] = arith.select {{%.*}}, [[VALID_SIZE]], [[C0]] : index
  // CHECK: [[BOUNDARY_GM:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[GM_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  // CHECK: [[BOUNDARY_LOCAL:%.*]] = memref.reinterpret_cast [[VALUE_BUFFER]] to offset: {{.}}[[C0]]{{.}}, sizes: [4], strides: [1] : memref<4xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: [[GM_SUBVIEW:%.*]] = memref.subview [[BOUNDARY_GM]][0] {{.}}[[TRANSFER_SIZE]]{{.}} [1]
  // CHECK: [[LOCAL_SUBVIEW:%.*]] = memref.subview [[BOUNDARY_LOCAL]]{{.*}}{{.}}[[TRANSFER_SIZE]]{{.}} [1]
  // CHECK: memref.copy [[LOCAL_SUBVIEW]], [[GM_SUBVIEW]]
  // CHECK-NOT: memref.store
  // CHECK: tt.return

  tt.func public @gather_padding(%base: !tv.ptr<f32>, %size: index, %stride: index, %indices: tensor<4xi32>) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
    %view = tv.make_gather_scatter_view %base_view : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.gather_scatter_view<tile = [4], sparse_dim = [0], padding_value = nan>>
    %result = tv.view_load %view[%indices] : !tv.tensor_view<?xf32, strides=[?], #tv.gather_scatter_view<tile = [4], sparse_dim = [0], padding_value = nan>>, tensor<4xi32> -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @gather_padding(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDICES:%.*]]: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK-DAG: [[NAN:%.*]] = arith.constant 0x7FC00000 : f32
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[TRUE:%.*]] = arith.constant true
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: [[ALL_IN_BOUNDS:%.*]] = scf.for [[CHECK_IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args([[VALID:%.*]] = [[TRUE]]) -> (i1) {
  // CHECK: [[RAW_INDEX:%.*]] = tensor.extract [[INDICES]][[[CHECK_IV]]] : tensor<4xi32>
  // CHECK: [[INDEX:%.*]] = arith.index_cast [[RAW_INDEX]] : i32 to index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[INDEX]], [[C0]] : index
  // CHECK: [[BELOW_EXTENT:%.*]] = arith.cmpi slt, [[INDEX]], [[SIZE]] : index
  // CHECK: [[INDEX_IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[BELOW_EXTENT]] : i1
  // CHECK: [[NEXT_VALID:%.*]] = arith.andi [[VALID]], [[INDEX_IN_BOUNDS]] : i1
  // CHECK: scf.yield [[NEXT_VALID]] : i1
  // CHECK: [[RESULT:%.*]] = scf.if [[ALL_IN_BOUNDS]] -> (tensor<4xf32>) {
  // CHECK: [[EMPTY:%.*]] = tensor.empty() : tensor<4xf32>
  // CHECK: [[GATHERED:%.*]] = scf.for [[IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args([[ACC:%.*]] = [[EMPTY]]) -> (tensor<4xf32>) {
  // CHECK: [[RAW_COORDINATE:%.*]] = tensor.extract [[INDICES]][[[IV]]] : tensor<4xi32>
  // CHECK: [[COORDINATE:%.*]] = arith.index_cast [[RAW_COORDINATE]] : i32 to index
  // CHECK: [[ELEMENT_NON_NEGATIVE:%.*]] = arith.cmpi sge, [[COORDINATE]], [[C0]] : index
  // CHECK: [[ELEMENT_BELOW_EXTENT:%.*]] = arith.cmpi slt, [[COORDINATE]], [[SIZE]] : index
  // CHECK: [[ELEMENT_IN_BOUNDS:%.*]] = arith.andi [[ELEMENT_NON_NEGATIVE]], [[ELEMENT_BELOW_EXTENT]] : i1
  // CHECK: [[ELEMENT_OFFSET:%.*]] = arith.muli [[COORDINATE]], [[STRIDE]] : index
  // CHECK: [[ELEMENT:%.*]] = scf.if [[ELEMENT_IN_BOUNDS]] -> (f32) {
  // CHECK: [[GM_ELEMENT:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[ELEMENT_OFFSET]]{{.}}, sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK: [[LOADED:%.*]] = memref.load [[GM_ELEMENT]][[[C0]]] : memref<1xf32, strided<[1], offset: ?>>
  // CHECK: scf.yield [[LOADED]] : f32
  // CHECK: } else {
  // CHECK: scf.yield [[NAN]] : f32
  // CHECK: [[INSERTED:%.*]] = tensor.insert [[ELEMENT]] into [[ACC]][[[IV]]] : tensor<4xf32>
  // CHECK: scf.yield [[INSERTED]] : tensor<4xf32>
  // CHECK: scf.yield [[GATHERED]] : tensor<4xf32>
  // CHECK: } else {
  // CHECK: [[FALLBACK_EMPTY:%.*]] = tensor.empty() : tensor<4xf32>
  // CHECK: [[FALLBACK:%.*]] = scf.for [[FALLBACK_IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args([[FALLBACK_ACC:%.*]] = [[FALLBACK_EMPTY]]) -> (tensor<4xf32>) {
  // CHECK: [[FALLBACK_RAW_COORDINATE:%.*]] = tensor.extract [[INDICES]][[[FALLBACK_IV]]] : tensor<4xi32>
  // CHECK: [[FALLBACK_COORDINATE:%.*]] = arith.index_cast [[FALLBACK_RAW_COORDINATE]] : i32 to index
  // CHECK: [[FALLBACK_OFFSET:%.*]] = arith.muli [[FALLBACK_COORDINATE]], [[STRIDE]] : index
  // CHECK: [[FALLBACK_ELEMENT:%.*]] = scf.if {{%.*}} -> (f32) {
  // CHECK: [[FALLBACK_GM_ELEMENT:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[FALLBACK_OFFSET]]{{.}}, sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK: [[FALLBACK_LOADED:%.*]] = memref.load [[FALLBACK_GM_ELEMENT]][[[C0]]] : memref<1xf32, strided<[1], offset: ?>>
  // CHECK: scf.yield [[FALLBACK_LOADED]] : f32
  // CHECK: } else {
  // CHECK: scf.yield [[NAN]] : f32
  // CHECK: [[FALLBACK_INSERTED:%.*]] = tensor.insert [[FALLBACK_ELEMENT]] into [[FALLBACK_ACC]][[[FALLBACK_IV]]] : tensor<4xf32>
  // CHECK: scf.yield [[FALLBACK_INSERTED]] : tensor<4xf32>
  // CHECK: scf.yield [[FALLBACK]] : tensor<4xf32>
  // CHECK: tt.return [[RESULT]] : tensor<4xf32>

  tt.func public @gather_block_padding(%base: !tv.ptr<f32>, %rows: index, %columns: index, %row_stride: index, %row_indices: tensor<2xi32>, %column_block: index) -> tensor<2x4xf32> {
    %c1 = arith.constant 1 : index
    %base_view = tv.make_tensor_view %base, sizes = [%rows, %columns], strides = [%row_stride, %c1] : !tv.ptr<f32> -> !tv.tensor_view<?x?xf32, strides=[?, 1]>
    %view = tv.make_gather_scatter_view %base_view : !tv.tensor_view<?x?xf32, strides=[?, 1]> -> !tv.tensor_view<?x?xf32, strides=[?, 1], #tv.gather_scatter_view<tile = [2, 4], sparse_dim = [0], padding_value = zero>>
    %result = tv.view_load %view[%row_indices, %column_block] : !tv.tensor_view<?x?xf32, strides=[?, 1], #tv.gather_scatter_view<tile = [2, 4], sparse_dim = [0], padding_value = zero>>, tensor<2xi32>, index -> tensor<2x4xf32>
    tt.return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: tt.func public @gather_block_padding(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[ROWS:%.*]]: index, [[COLUMNS:%.*]]: index, [[ROW_STRIDE:%.*]]: index, [[ROW_INDICES:%.*]]: tensor<2xi32>, [[COLUMN_BLOCK:%.*]]: index) -> tensor<2x4xf32> {
  // CHECK-DAG: [[PADDING:%.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C2:%.*]] = arith.constant 2 : index
  // Every sparse index is range checked before a transfer is chosen.
  // CHECK: [[ALL_IN_BOUNDS:%.*]] = scf.for {{%.*}} = [[C0]] to [[C2]] step [[C1]] iter_args({{%.*}} = {{%.*}}) -> (i1) {
  // CHECK: [[RESULT:%.*]] = scf.if {{%.*}} -> (tensor<2x4xf32>) {
  // One tile-sized buffer is allocated ahead of the gather loop and every
  // sparse row lands in its own window, so no tensor is carried across
  // iterations.
  // CHECK: [[IN_BOUNDS_BUFFER:%.*]] = memref.alloc() : memref<2x4xf32>
  // CHECK: scf.for [[IN_BOUNDS_IV:%.*]] = [[C0]] to [[C2]] step [[C1]] {
  // CHECK: [[IN_BOUNDS_SLICE_OFFSET:%.*]] = arith.muli [[IN_BOUNDS_IV]], [[C4]] : index
  // CHECK: tensor.extract [[ROW_INDICES]][[[IN_BOUNDS_IV]]] {DiscreteMemAccess} : tensor<2xi32>
  // CHECK: [[IN_BOUNDS_GM:%.*]] = memref.reinterpret_cast [[BASE]]{{.*}}sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: [[IN_BOUNDS_LOCAL:%.*]] = memref.reinterpret_cast [[IN_BOUNDS_BUFFER]] to offset: {{.}}[[IN_BOUNDS_SLICE_OFFSET]]{{.}}, sizes: [4], strides: [1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: memref.copy [[IN_BOUNDS_GM]], [[IN_BOUNDS_LOCAL]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: }
  // CHECK: [[IN_BOUNDS_RESULT:%.*]] = bufferization.to_tensor [[IN_BOUNDS_BUFFER]] restrict : memref<2x4xf32> to tensor<2x4xf32>
  // CHECK: scf.yield [[IN_BOUNDS_RESULT]] : tensor<2x4xf32>
  // CHECK: } else {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<2x4xf32>
  // CHECK: linalg.fill ins([[PADDING]] : f32) outs([[BUFFER]] : memref<2x4xf32>)
  // CHECK: scf.for [[IV:%.*]] = [[C0]] to [[C2]] step [[C1]] {
  // CHECK: [[SLICE_OFFSET:%.*]] = arith.muli [[IV]], [[C4]] : index
  // CHECK: [[RAW_ROW:%.*]] = tensor.extract [[ROW_INDICES]][[[IV]]] {DiscreteMemAccess} : tensor<2xi32>
  // CHECK: [[ROW:%.*]] = arith.index_cast [[RAW_ROW]] : i32 to index
  // CHECK: arith.maxsi [[ROW]], [[C0]] : index
  // CHECK: arith.muli [[COLUMN_BLOCK]], [[C4]] : index
  // CHECK: [[GM_BLOCK:%.*]] = memref.reinterpret_cast [[BASE]]{{.*}}sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: [[LOCAL_BLOCK:%.*]] = memref.reinterpret_cast [[BUFFER]] to offset: {{.}}[[SLICE_OFFSET]]{{.}}, sizes: [4], strides: [1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: [[GM_SUBVIEW:%.*]] = memref.subview [[GM_BLOCK]][0] {{.*}} [1] : memref<4xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK: [[LOCAL_SUBVIEW:%.*]] = memref.subview [[LOCAL_BLOCK]]{{.*}} : memref<4xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK: memref.copy [[GM_SUBVIEW]], [[LOCAL_SUBVIEW]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK: }
  // CHECK: [[BOUNDARY_RESULT:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<2x4xf32> to tensor<2x4xf32>
  // CHECK: scf.yield [[BOUNDARY_RESULT]] : tensor<2x4xf32>
  // CHECK-NOT: memref.load
  // CHECK-NOT: tensor.insert_slice
  // CHECK: tt.return [[RESULT]] : tensor<2x4xf32>

  tt.func public @gather_unit_dims_padding(%base: !tv.ptr<f32>, %size0: index, %size1: index, %size2: index, %size3: index, %stride0: index, %stride1: index, %stride2: index, %indices: tensor<2xi32>, %block1: index, %index2: index, %block3: index) -> tensor<2x4x1x8xf32> {
    %c1 = arith.constant 1 : index
    %base_view = tv.make_tensor_view %base, sizes = [%size0, %size1, %size2, %size3], strides = [%stride0, %stride1, %stride2, %c1] : !tv.ptr<f32> -> !tv.tensor_view<?x?x?x?xf32, strides=[?, ?, ?, 1]>
    %view = tv.make_gather_scatter_view %base_view : !tv.tensor_view<?x?x?x?xf32, strides=[?, ?, ?, 1]> -> !tv.tensor_view<?x?x?x?xf32, strides=[?, ?, ?, 1], #tv.gather_scatter_view<tile = [2, 4, 1, 8], sparse_dim = [0], padding_value = zero>>
    %result = tv.view_load %view[%indices, %block1, %index2, %block3] : !tv.tensor_view<?x?x?x?xf32, strides=[?, ?, ?, 1], #tv.gather_scatter_view<tile = [2, 4, 1, 8], sparse_dim = [0], padding_value = zero>>, tensor<2xi32>, index, index, index -> tensor<2x4x1x8xf32>
    tt.return %result : tensor<2x4x1x8xf32>
  }

  // Unit tile dimensions stay out of the DMA rank while the local buffer keeps
  // the full logical tile shape.
  // CHECK-LABEL: tt.func public @gather_unit_dims_padding(
  // CHECK: [[RESULT:%.*]] = scf.if {{%.*}} -> (tensor<2x4x1x8xf32>) {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<2x4x1x8xf32>
  // CHECK: scf.for [[IV:%.*]] = {{%.*}} to {{%.*}} step {{%.*}} {
  // CHECK: [[GM:%.*]] = memref.reinterpret_cast {{.*}} sizes: [4, 8], strides: {{.*}} : memref<?xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>
  // CHECK: [[LOCAL:%.*]] = memref.reinterpret_cast [[BUFFER]]{{.*}}sizes: [4, 8], strides: [8, 1] : memref<2x4x1x8xf32> to memref<4x8xf32, strided<[8, 1], offset: ?>>
  // CHECK: memref.copy [[GM]], [[LOCAL]]
  // CHECK: }
  // CHECK: [[GATHERED:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<2x4x1x8xf32> to tensor<2x4x1x8xf32>
  // CHECK: scf.yield [[GATHERED]] : tensor<2x4x1x8xf32>
  // CHECK: } else {
  // CHECK: [[BOUNDARY_BUFFER:%.*]] = memref.alloc() : memref<2x4x1x8xf32>
  // CHECK: linalg.fill {{.*}} outs([[BOUNDARY_BUFFER]] : memref<2x4x1x8xf32>)
  // CHECK: scf.for
  // CHECK: [[BOUNDARY_GM:%.*]] = memref.reinterpret_cast {{.*}} sizes: [4, 8], strides: {{.*}} : memref<?xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>
  // CHECK: memref.copy
  // CHECK: }
  // CHECK: bufferization.to_tensor [[BOUNDARY_BUFFER]] restrict : memref<2x4x1x8xf32> to tensor<2x4x1x8xf32>
  // CHECK-NOT: memref.load
  // CHECK-NOT: tensor.insert_slice
  // CHECK: tt.return [[RESULT]] : tensor<2x4x1x8xf32>

  tt.func public @multi_sparse_block_padding(%base: !tv.ptr<f32>, %size0: index, %size1: index, %size2: index, %stride0: index, %stride1: index, %index0: tensor<2xi32>, %index1: tensor<3xi32>, %block2: index) -> tensor<2x3x4xf32> {
    %c1 = arith.constant 1 : index
    %base_view = tv.make_tensor_view %base, sizes = [%size0, %size1, %size2], strides = [%stride0, %stride1, %c1] : !tv.ptr<f32> -> !tv.tensor_view<?x?x?xf32, strides=[?, ?, 1]>
    %view = tv.make_gather_scatter_view %base_view : !tv.tensor_view<?x?x?xf32, strides=[?, ?, 1]> -> !tv.tensor_view<?x?x?xf32, strides=[?, ?, 1], #tv.gather_scatter_view<tile = [2, 3, 4], sparse_dim = [0, 1], padding_value = zero>>
    %result = tv.view_load %view[%index0, %index1, %block2] : !tv.tensor_view<?x?x?xf32, strides=[?, ?, 1], #tv.gather_scatter_view<tile = [2, 3, 4], sparse_dim = [0, 1], padding_value = zero>>, tensor<2xi32>, tensor<3xi32>, index -> tensor<2x3x4xf32>
    tt.return %result : tensor<2x3x4xf32>
  }

  // Two sparse dimensions nest two loops around a single tile buffer.
  // CHECK-LABEL: tt.func public @multi_sparse_block_padding(
  // CHECK: [[RESULT:%.*]] = scf.if {{%.*}} -> (tensor<2x3x4xf32>) {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<2x3x4xf32>
  // CHECK: scf.for [[IV0:%.*]] = {{%.*}} to {{%.*}} step {{%.*}} {
  // CHECK: scf.for [[IV1:%.*]] = {{%.*}} to {{%.*}} step {{%.*}} {
  // CHECK: [[GM:%.*]] = memref.reinterpret_cast {{.*}} sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: [[LOCAL:%.*]] = memref.reinterpret_cast [[BUFFER]]{{.*}}sizes: [4], strides: [1] : memref<2x3x4xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: memref.copy [[GM]], [[LOCAL]]
  // CHECK: }
  // CHECK: }
  // CHECK: [[GATHERED:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<2x3x4xf32> to tensor<2x3x4xf32>
  // CHECK: scf.yield [[GATHERED]] : tensor<2x3x4xf32>
  // CHECK: } else {
  // CHECK: [[BOUNDARY_BUFFER:%.*]] = memref.alloc() : memref<2x3x4xf32>
  // CHECK: linalg.fill {{.*}} outs([[BOUNDARY_BUFFER]] : memref<2x3x4xf32>)
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: memref.copy
  // CHECK: }
  // CHECK: }
  // CHECK: bufferization.to_tensor [[BOUNDARY_BUFFER]] restrict : memref<2x3x4xf32> to tensor<2x3x4xf32>
  // CHECK-NOT: memref.load
  // CHECK-NOT: tensor.insert_slice
  // CHECK: tt.return [[RESULT]] : tensor<2x3x4xf32>

  tt.func public @multi_sparse_block_store(%base: !tv.ptr<f32>, %size0: index, %size1: index, %size2: index, %stride0: index, %stride1: index, %index0: tensor<2xi32>, %index1: tensor<3xi32>, %block2: index, %value: tensor<2x3x4xf32>) {
    %c1 = arith.constant 1 : index
    %base_view = tv.make_tensor_view %base, sizes = [%size0, %size1, %size2], strides = [%stride0, %stride1, %c1] : !tv.ptr<f32> -> !tv.tensor_view<?x?x?xf32, strides=[?, ?, 1]>
    %view = tv.make_gather_scatter_view %base_view : !tv.tensor_view<?x?x?xf32, strides=[?, ?, 1]> -> !tv.tensor_view<?x?x?xf32, strides=[?, ?, 1], #tv.gather_scatter_view<tile = [2, 3, 4], sparse_dim = [0, 1], padding_value = zero>>
    tv.view_store %view[%index0, %index1, %block2], %value : !tv.tensor_view<?x?x?xf32, strides=[?, ?, 1], #tv.gather_scatter_view<tile = [2, 3, 4], sparse_dim = [0, 1], padding_value = zero>>, tensor<2x3x4xf32>, tensor<2xi32>, tensor<3xi32>, index
    tt.return
  }

  // CHECK-LABEL: tt.func public @multi_sparse_block_store(
  // CHECK: scf.if {{%.*}} {
  // CHECK: [[VALUE_BUFFER:%.*]] = bufferization.to_{{memref|buffer}} {{%.*}} : tensor<2x3x4xf32> to memref<2x3x4xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: [[LOCAL:%.*]] = memref.reinterpret_cast [[VALUE_BUFFER]]{{.*}}sizes: [4], strides: [1] : memref<2x3x4xf32> to memref<4xf32, strided<[1], offset: ?>>
  // CHECK: memref.copy [[LOCAL]], {{%.*}}
  // CHECK: }
  // CHECK: }
  // CHECK: } else {
  // CHECK: [[BOUNDARY_VALUE_BUFFER:%.*]] = bufferization.to_{{memref|buffer}} {{%.*}} : tensor<2x3x4xf32> to memref<2x3x4xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: memref.copy {{%.*}}, {{%.*}}
  // CHECK: }
  // CHECK: }
  // CHECK-NOT: memref.store
  // CHECK: tt.return

  tt.func public @negative_infinity_padding(%base: !tv.ptr<f32>, %size: index, %stride: index, %index: index) -> tensor<4xf32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
    %view = tv.make_partition_view %base_view : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = -inf>>
    %result = tv.view_load %view[%index] : !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = -inf>>, index -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @negative_infinity_padding(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xf32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDEX:%.*]]: index) -> tensor<4xf32> {
  // CHECK-DAG: [[NEGATIVE_INFINITY:%.*]] = arith.constant 0xFF800000 : f32
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: [[ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[ORIGIN]], [[C0]] : index
  // CHECK: [[END:%.*]] = arith.addi [[ORIGIN]], [[C4]] : index
  // CHECK: [[WITHIN_EXTENT:%.*]] = arith.cmpi sle, [[END]], [[SIZE]] : index
  // CHECK: [[IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[WITHIN_EXTENT]] : i1
  // CHECK: [[RESULT:%.*]] = scf.if [[IN_BOUNDS]] -> (tensor<4xf32>) {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<4xf32>
  // CHECK: [[TILE_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[TILE_OFFSET:%.*]] = arith.muli [[TILE_ORIGIN]], [[STRIDE]] : index
  // CHECK: [[GM_TILE:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[TILE_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  // CHECK: memref.copy [[GM_TILE]], [[BUFFER]] : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32>
  // CHECK: [[TENSOR:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<4xf32> to tensor<4xf32>
  // CHECK: scf.yield [[TENSOR]] : tensor<4xf32>
  // CHECK: } else {
  // CHECK: [[BOUNDARY_BUFFER:%.*]] = memref.alloc() : memref<4xf32>
  // CHECK: linalg.fill ins([[NEGATIVE_INFINITY]] : f32) outs([[BOUNDARY_BUFFER]] : memref<4xf32>)
  // CHECK: memref.copy
  // CHECK-NOT: memref.load
  // CHECK: tt.return [[RESULT]] : tensor<4xf32>

  tt.func public @integer_zero_padding(%base: !tv.ptr<i32>, %size: index, %stride: index, %index: index) -> tensor<4xi32> {
    %base_view = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<i32> -> !tv.tensor_view<?xi32, strides=[?]>
    %view = tv.make_partition_view %base_view : !tv.tensor_view<?xi32, strides=[?]> -> !tv.tensor_view<?xi32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>
    %result = tv.view_load %view[%index] : !tv.tensor_view<?xi32, strides=[?], #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>>, index -> tensor<4xi32>
    tt.return %result : tensor<4xi32>
  }

  // CHECK-LABEL: tt.func public @integer_zero_padding(
  // CHECK-SAME: [[BASE:%.*]]: memref<?xi32>, [[SIZE:%.*]]: index, [[STRIDE:%.*]]: index, [[INDEX:%.*]]: index) -> tensor<4xi32> {
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0 : i32
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: [[ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[NON_NEGATIVE:%.*]] = arith.cmpi sge, [[ORIGIN]], [[C0]] : index
  // CHECK: [[END:%.*]] = arith.addi [[ORIGIN]], [[C4]] : index
  // CHECK: [[WITHIN_EXTENT:%.*]] = arith.cmpi sle, [[END]], [[SIZE]] : index
  // CHECK: [[IN_BOUNDS:%.*]] = arith.andi [[NON_NEGATIVE]], [[WITHIN_EXTENT]] : i1
  // CHECK: [[RESULT:%.*]] = scf.if [[IN_BOUNDS]] -> (tensor<4xi32>) {
  // CHECK: [[BUFFER:%.*]] = memref.alloc() : memref<4xi32>
  // CHECK: [[TILE_ORIGIN:%.*]] = arith.muli [[INDEX]], [[C4]] : index
  // CHECK: [[TILE_OFFSET:%.*]] = arith.muli [[TILE_ORIGIN]], [[STRIDE]] : index
  // CHECK: [[GM_TILE:%.*]] = memref.reinterpret_cast [[BASE]] to offset: {{.}}[[TILE_OFFSET]]{{.}}, sizes: [4], strides: {{.}}[[STRIDE]]{{.}} : memref<?xi32> to memref<4xi32, strided<[?], offset: ?>>
  // CHECK: memref.copy [[GM_TILE]], [[BUFFER]] : memref<4xi32, strided<[?], offset: ?>> to memref<4xi32>
  // CHECK: [[TENSOR:%.*]] = bufferization.to_tensor [[BUFFER]] restrict : memref<4xi32> to tensor<4xi32>
  // CHECK: scf.yield [[TENSOR]] : tensor<4xi32>
  // CHECK: } else {
  // CHECK: [[BOUNDARY_BUFFER:%.*]] = memref.alloc() : memref<4xi32>
  // CHECK: linalg.fill ins([[ZERO]] : i32) outs([[BOUNDARY_BUFFER]] : memref<4xi32>)
  // CHECK: memref.copy
  // CHECK-NOT: memref.load
  // CHECK: tt.return [[RESULT]] : tensor<4xi32>
}
