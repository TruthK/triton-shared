# UKernelOpInterfaceGen
set(LLVM_TARGET_DEFINITIONS UKernelOpInterface.td)
mlir_tablegen(UKernelOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(UKernelOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(UKernelOpInterfaceGen)

# PartitionableLoopsInterfaceGen
set(LLVM_TARGET_DEFINITIONS PartitionableLoopsInterface.td)
mlir_tablegen(PartitionableLoopsInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(PartitionableLoopsInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(PartitionableLoopsInterfaceGen)