# IREEVectorExtDialectGen
set(LLVM_TARGET_DEFINITIONS VectorExtBase.td)
mlir_tablegen(VectorExtDialect.h.inc -gen-dialect-decls)
mlir_tablegen(VectorExtDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(IREEVectorExtDialectGen)

# IREEVectorExtEnumsGen
set(LLVM_TARGET_DEFINITIONS VectorExtAttrs.td)
mlir_tablegen(VectorExtEnums.h.inc -gen-enum-decls)
mlir_tablegen(VectorExtEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(IREEVectorExtEnumsGen)

# IREEVectorExtAttrsGen
set(LLVM_TARGET_DEFINITIONS VectorExtAttrs.td)
mlir_tablegen(VectorExtAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(VectorExtAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(IREEVectorExtAttrsGen)

# IREEVectorExtInterfacesGen
set(LLVM_TARGET_DEFINITIONS VectorExtInterfaces.td)
mlir_tablegen(VectorExtAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(VectorExtAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(IREEVectorExtInterfacesGen)

# IREEVectorExtOpsGen
set(LLVM_TARGET_DEFINITIONS VectorExtOps.td)
mlir_tablegen(VectorExtOps.h.inc -gen-op-decls)
mlir_tablegen(VectorExtOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(IREEVectorExtOpsGen)
