// RUN: triton-shared-opt --tensor-view-lowering %s | FileCheck %s

module {
  tt.func public @ptr_load(%pointers: tensor<8x8x!tt.ptr<f32>>, %index0: tensor<4xindex>, %index1: tensor<4xindex>) -> tensor<4xf32> {
    %result = tv.ptr_load %pointers, indices = [%index0, %index1] : tensor<8x8x!tt.ptr<f32>>, tensor<4xindex>, tensor<4xindex> -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @ptr_load(
  // CHECK-SAME: [[POINTERS:%.*]]: tensor<8x8x!tt.ptr<f32>>,
  // CHECK-SAME: [[INDEX0:%.*]]: tensor<4xindex>,
  // CHECK-SAME: [[INDEX1:%.*]]: tensor<4xindex>) -> tensor<4xf32> {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK-DAG: [[EMPTY:%.*]] = tensor.empty() : tensor<4xf32>
  // CHECK: [[RESULT:%.*]] = scf.for [[IV:%.*]] = [[C0]] to [[C4]] step [[C1]] iter_args([[ACC:%.*]] = [[EMPTY]]) -> (tensor<4xf32>) {
  // CHECK: [[COORD0:%.*]] = tensor.extract [[INDEX0]][[[IV]]] : tensor<4xindex>
  // CHECK: [[COORD1:%.*]] = tensor.extract [[INDEX1]][[[IV]]] : tensor<4xindex>
  // CHECK: [[POINTER:%.*]] = tensor.extract [[POINTERS]][[[COORD0]], [[COORD1]]] : tensor<8x8x!tt.ptr<f32>>
  // CHECK: [[VALUE:%.*]] = tt.load [[POINTER]] {DiscreteMemAccess} : !tt.ptr<f32>
  // CHECK: [[INSERTED:%.*]] = tensor.insert [[VALUE]] into [[ACC]][[[IV]]] : tensor<4xf32>
  // CHECK: scf.yield {DiscreteMemAccess} [[INSERTED]] : tensor<4xf32>
  // CHECK: } {ExtractedLoadOrStore}
  // CHECK: tt.return [[RESULT]] : tensor<4xf32>

  tt.func public @ptr_store(%pointers: tensor<8x8x!tt.ptr<f32>>, %index0: tensor<4xindex>, %index1: tensor<4xindex>, %value: tensor<4xf32>) {
    tv.ptr_store %pointers, %value, indices = [%index0, %index1] : tensor<8x8x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xindex>, tensor<4xindex>
    tt.return
  }

  // CHECK-LABEL: tt.func public @ptr_store(
  // CHECK-SAME: [[POINTERS:%.*]]: tensor<8x8x!tt.ptr<f32>>,
  // CHECK-SAME: [[INDEX0:%[^ ,]*]]: tensor<4xindex>,
  // CHECK-SAME: [[INDEX1:%.*]]: tensor<4xindex>,
  // CHECK-SAME: [[VALUE:%.*]]: tensor<4xf32>) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C4:%.*]] = arith.constant 4 : index
  // CHECK: scf.for [[IV:%.*]] = [[C0]] to [[C4]] step [[C1]] {
  // CHECK: [[COORD0:%.*]] = tensor.extract [[INDEX0]][[[IV]]] : tensor<4xindex>
  // CHECK: [[COORD1:%.*]] = tensor.extract [[INDEX1]][[[IV]]] : tensor<4xindex>
  // CHECK: [[POINTER:%.*]] = tensor.extract [[POINTERS]][[[COORD0]], [[COORD1]]] : tensor<8x8x!tt.ptr<f32>>
  // CHECK: [[ELEMENT:%.*]] = tensor.extract [[VALUE]][[[IV]]] : tensor<4xf32>
  // CHECK: tt.store [[POINTER]], [[ELEMENT]] {DiscreteMemAccess} : !tt.ptr<f32>
  // CHECK: } {ExtractedLoadOrStore}
  // CHECK: tt.return
}
