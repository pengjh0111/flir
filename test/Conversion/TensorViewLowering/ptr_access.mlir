// RUN: triton-shared-opt --tensor-view-lowering %s | FileCheck %s

module {
  tt.func public @ptr_load(%pointers: tensor<8x8x!tt.ptr<f32>>, %index0: tensor<4xindex>, %index1: tensor<4xindex>) -> tensor<4xf32> {
    %result = tv.ptr_load %pointers, indices = [%index0, %index1] : tensor<8x8x!tt.ptr<f32>>, tensor<4xindex>, tensor<4xindex> -> tensor<4xf32>
    tt.return %result : tensor<4xf32>
  }

  tt.func public @ptr_store(%pointers: tensor<8x8x!tt.ptr<f32>>, %index0: tensor<4xindex>, %index1: tensor<4xindex>, %value: tensor<4xf32>) {
    tv.ptr_store %pointers, %value, indices = [%index0, %index1] : tensor<8x8x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xindex>, tensor<4xindex>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @ptr_load(
// CHECK: scf.for
// CHECK-COUNT-3: tensor.extract
// CHECK: tt.load {{.*}} {DiscreteMemAccess}
// CHECK: tensor.insert
// CHECK: scf.yield {DiscreteMemAccess}
// CHECK: } {ExtractedLoadOrStore}
// CHECK-NOT: tv.
// CHECK: tt.return

// CHECK-LABEL: tt.func public @ptr_store(
// CHECK: scf.for
// CHECK-COUNT-4: tensor.extract
// CHECK: tt.store {{.*}} {DiscreteMemAccess}
// CHECK: } {ExtractedLoadOrStore}
// CHECK-NOT: tv.
// CHECK: tt.return
