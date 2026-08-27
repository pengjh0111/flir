set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Ops tablegen
set(LLVM_TARGET_DEFINITIONS CommonIROps.td)
mlir_tablegen(CommonIRDialect.h.inc -gen-dialect-decls -dialect=tile)
mlir_tablegen(CommonIRDialect.cpp.inc -gen-dialect-defs -dialect=tile)
mlir_tablegen(CommonIROps.h.inc -gen-op-decls)
mlir_tablegen(CommonIROps.cpp.inc -gen-op-defs)
add_mlir_doc(CommonIRDialect CommonIRDialect dialects/ -gen-dialect-doc)
add_mlir_doc(CommonIROps CommonIROps dialects/ -gen-op-doc)
add_public_tablegen_target(CommonIRTableGen)

# AttrDefs tablegen
set(LLVM_TARGET_DEFINITIONS CommonIRAttrDefs.td)
mlir_tablegen(CommonIROpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(CommonIROpsAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(CommonIREnums.h.inc -gen-enum-decls)
mlir_tablegen(CommonIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CommonIRAttrDefsIncGen)

# Types tablegen
set(LLVM_TARGET_DEFINITIONS CommonIRTypes.td)
mlir_tablegen(CommonIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CommonIRTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(CommonIRTypesIncGen)