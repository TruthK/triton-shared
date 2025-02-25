#ifndef LINALG_TO_LLVM_CONVERSION_PASSES
#define LINALG_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToLLVMPass : Pass<"linalg-to-llvm", "mlir::ModuleOp"> {
  let summary = "An pipeline to lower the main dialects (arith, linalg, memref, scf, vector) down to LLVM.";
}

def MemrefCopyToLinalgPass :
    InterfacePass<"memrefcopy-to-linalg", "mlir::FunctionOpInterface"> {
  let summary = "Convert memref.copy to linalg op";
}

#endif
