#ifndef STRUCTURED_TO_MEMREF_CONVERSION_PASSES
#define STRUCTURED_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
}
def SimplifyTTSMakeTPtr : Pass<"simplify-tts-make-tptr","mlir::ModuleOp"> {
  let summary = "Simplify tts.make_tptr operations by analyzing mask chains";
  let description = [{
    Analyzes the use chain of tts.make_tptr operations to detect cases where
    tensor dimensions can be safely optimized based on mask computation patterns.
    Specifically looks for patterns involving arith.min, arith.max, and offset
    calculations to eliminate redundant shape checks.
  }];
  
}
#endif
