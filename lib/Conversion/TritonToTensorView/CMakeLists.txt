add_triton_library(TritonToTensorView
  TensorViewBuilder.cpp
  TritonToTensorView.cpp

  DEPENDS
  TritonToTensorViewConversionPassIncGen

  LINK_LIBS
  TensorViewIR
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRArithDialect
  MLIRBufferizationDialect
  TritonIR
)
