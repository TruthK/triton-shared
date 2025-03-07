# IREEGPUDialectGen
set(LLVM_TARGET_DEFINITIONS IREEGPUDialect.td)
mlir_tablegen(IREEGPUDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IREEGPUDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(IREEGPUDialectGen)

# IREEGPUEnums
set(LLVM_TARGET_DEFINITIONS IREEGPUEnums.td)
mlir_tablegen(IREEGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(IREEGPUEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(IREEGPUEnums)

# IREEGPUAttrs
set(LLVM_TARGET_DEFINITIONS IREEGPUAttrs.td)
mlir_tablegen(IREEGPUAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(IREEGPUAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(IREEGPUAttrs)

# IREEGPUInterfaces
set(LLVM_TARGET_DEFINITIONS IREEGPUInterfaces.td)
mlir_tablegen(IREEGPUInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(IREEGPUInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(IREEGPUInterfaces)

# IREEGPUOpsGen
set(LLVM_TARGET_DEFINITIONS IREEGPUOps.td)
mlir_tablegen(IREEGPUOps.h.inc -gen-op-decls)
mlir_tablegen(IREEGPUOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(IREEGPUOpsGen)