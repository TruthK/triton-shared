//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES
#define TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def TritonToLinalgExperimental : Pass<"triton-to-linalg-experimental", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let constructor = "triton::createTritonToLinalgExperimentalPass()";
}

def ConvertTritonStructuredToMemref : Pass<"convert-triton-structured-to-memref", "mlir::func::FuncOp"> {
  let summary = "Convert Triton Structured ops to memref ops";
  let description = [{
    This pass converts Triton Structured transfer ops to memref transfer ops.
  }];
  let constructor = "triton::createConvertTritonStructuredToMemrefPass()";
}

def LinalgGenericFusionPass : Pass<"linalg-generic-fusion", "mlir::ModuleOp"> {
  let summary = "Fuse adjacent linalg.generic operations";
  let description = [{
    This pass fuses adjacent linalg.generic operations into a single operation
    when possible to reduce intermediate memory usage and improve performance.
  }];
  let constructor = "triton::createLinalgGenericFusionPass()";
}

def ConvertTTSTransferOp : Pass<"convert-triton-transfer-op", "mlir::func::FuncOp"> {
  let summary = "Convert Triton Structured transfer ops to memref or vector transfer ops";
  let description = [{
    This pass converts Triton Structured transfer operations to either IREE vector extension 
    operations or vector dialect operations.
  }];
  let constructor = "triton::createConvertTTSTransferOpPass()";
  let options = [
    Option<"isTensorToVector", "tensor-to-vector", "bool", /*default=*/"false",
           "Convert TTS transfer ops directly to vector transfer operations">
  ];
}

def TTSLoopUnroll : Pass<"tts-loop-unroll", "mlir::ModuleOp"> {
  let summary = "Loop unroller";
  let description = [{
    The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
    the loop should be unrolled.
  }];
  let constructor = "triton::createTTSLoopUnrollPass()";
}

#endif
