#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonArithToLinalg : Pass<"triton-arith-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Triton arithmetic operations into linalg";
  let options = [
      Option<"ttToFuncFunc", "tt-to-func-func", "bool", /*default*/"true",
             "Convert tt.func to func.func">,
      Option<"addptrToLinalg", "addptr-to-linalg", "bool", /*default*/"true",
             "Convert tt.addptr on tensors to linalg">,
      Option<"assertToCf", "assert-to-cf", "bool", /*default*/"true",
             "Convert tt.assert to cf.assert">,
  ];
}

#endif
