#ifndef TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES
#define TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToLLVM : Pass<"linalg-to-llvm", "mlir::ModuleOp"> {
  let summary = "An pipeline to lower the main dialects (arith, linalg, memref, scf, vector) down to LLVM.";
  let constructor = "mlir::tts::createLinalgToLLVMPass()";
}

#endif
