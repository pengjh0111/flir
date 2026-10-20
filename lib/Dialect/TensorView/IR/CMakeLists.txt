# The IR is shared with AscendNPU-IR; frontend builders depend on Triton.
if(NOT TARGET TensorViewIR)
  add_mlir_dialect_library(TensorViewIR
    PARTIAL_SOURCES_INTENDED
    TensorViewDialect.cpp
    TensorViewTypes.cpp
    TensorViewAttrs.cpp
    TensorViewOps.cpp

    DEPENDS
    TensorViewTableGen
    TensorViewTypesIncGen
    TensorViewAttrsIncGen
    TensorViewInterfacesIncGen
    TensorViewTypeInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRDestinationStyleOpInterface
    MLIRSideEffectInterfaces
  )
endif()

if(COMMAND add_triton_library AND NOT TARGET TensorViewBuilder)
  add_triton_library(TensorViewBuilder
    TensorViewBuilder.cpp
    LINK_LIBS PUBLIC TensorViewIR MLIRArithDialect TritonIR)
endif()
