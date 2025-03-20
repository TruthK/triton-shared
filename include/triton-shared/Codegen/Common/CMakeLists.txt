add_subdirectory(GPU)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CommonCodegenPassIncGen)


# iree_tablegen_library(
#   NAME
#     FoldTensorExtractOpIncGen
#   TD_FILE
#     "FoldTensorExtractOp.td"
#   OUTS
#     --gen-rewriters FoldTensorExtractOp.cpp.inc
# )