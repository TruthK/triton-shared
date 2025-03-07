# EncodingEnumsGen
set(LLVM_TARGET_DEFINITIONS EncodingAttrs.td)
mlir_tablegen(EncodingEnums.h.inc -gen-enum-decls )
mlir_tablegen(EncodingEnums.cpp.inc -gen-enum-defs )
add_public_tablegen_target(EncodingEnumsGen)

# EncodingOpsIncGen
set(LLVM_TARGET_DEFINITIONS EncodingOps.td)
mlir_tablegen(EncodingOps.h.inc -gen-op-decls )
mlir_tablegen(EncodingOps.cpp.inc -gen-op-defs )
mlir_tablegen(EncodingDialect.h.inc -gen-dialect-decls -dialect=iree_encoding )
mlir_tablegen(EncodingDialect.cpp.inc -gen-dialect-defs -dialect=iree_encoding )
add_public_tablegen_target(EncodingOpsIncGen)

# EncodingInterfacesGen
set(LLVM_TARGET_DEFINITIONS EncodingInterfaces.td)
mlir_tablegen(EncodingInterfaces.h.inc -gen-attr-interface-decls )
mlir_tablegen(EncodingInterfaces.cpp.inc -gen-attr-interface-defs )
add_public_tablegen_target(EncodingInterfacesGen)

# EncodingTypesGen
set(LLVM_TARGET_DEFINITIONS EncodingAttrs.td)
mlir_tablegen(EncodingAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=iree_encoding )
mlir_tablegen(EncodingAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=iree_encoding )
mlir_tablegen(EncodingTypes.h.inc -gen-typedef-decls -typedefs-dialect=iree_encoding )
mlir_tablegen(EncodingTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=iree_encoding )
add_public_tablegen_target(EncodingTypesGen)
