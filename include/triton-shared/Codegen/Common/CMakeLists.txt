add_subdirectory(GPU)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CommonCodegenPassIncGen)


set(LLVM_TARGET_DEFINITIONS FoldTensorExtractOp.td)
mlir_tablegen(FoldTensorExtractOp.cpp.inc -gen-rewriters)
add_public_tablegen_target(FoldTensorExtractOpIncGen)

