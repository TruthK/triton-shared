set(LLVM_TARGET_DEFINITIONS IREECodegenAttrs.td)
mlir_tablegen(IREECodegenAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(IREECodegenAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(IREECodegenAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS IREECodegenDialect.td)
mlir_tablegen(IREECodegenDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IREECodegenDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(IREECodegenDialectIncGen)

set(LLVM_TARGET_DEFINITIONS IREECodegenOps.td)
mlir_tablegen(IREECodegenOps.h.inc -gen-op-decls)
mlir_tablegen(IREECodegenOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(IREECodegenOpsIncGen)

set(LLVM_TARGET_DEFINITIONS IREECodegenInterfaces.td)
mlir_tablegen(IREECodegenInterfaces.h.inc -gen-attrdef-decls -gen-attr-interface-decls)
mlir_tablegen(IREECodegenInterfaces.cpp.inc -gen-attrdef-defs -gen-attr-interface-defs)
add_public_tablegen_target(IREECodegenInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS UKernelOps.td)
mlir_tablegen(UKernelOps.h.inc -gen-op-decls)
mlir_tablegen(UKernelOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(UKernelOpsIncGen)
