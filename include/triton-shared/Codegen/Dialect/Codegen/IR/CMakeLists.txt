# IREECodegenDialectGen
set(LLVM_TARGET_DEFINITIONS IREECodegenDialect.td)
mlir_tablegen(IREECodegenDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IREECodegenDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(IREECodegenDialectGen)

# IREECodegenOpsGen
set(LLVM_TARGET_DEFINITIONS IREECodegenOps.td)
mlir_tablegen(IREECodegenOps.h.inc -gen-op-decls)
mlir_tablegen(IREECodegenOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(IREECodegenOpsGen)

# LoweringConfigInterfaceGen
set(LLVM_TARGET_DEFINITIONS IREECodegenInterfaces.td)
mlir_tablegen(IREECodegenInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(IREECodegenInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(LoweringConfigInterfaceGen)

# LoweringConfigGen
set(LLVM_TARGET_DEFINITIONS IREECodegenAttrs.td)
mlir_tablegen(IREECodegenAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(IREECodegenAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(LoweringConfigEnums.h.inc -gen-enum-decls)
mlir_tablegen(LoweringConfigEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(LoweringConfigGen)

# UKernelOpsGen
set(LLVM_TARGET_DEFINITIONS UKernelOps.td)
mlir_tablegen(UKernelOps.h.inc -gen-op-decls)
mlir_tablegen(UKernelOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(UKernelOpsGen)
