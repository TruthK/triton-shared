#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUDialect.h"

#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUDialect.cpp.inc"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUOps.h"

namespace mlir::tts::GPU {

void TTSGPUDialect::initialize() {
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUOps.cpp.inc"
      >();
}

} // namespace mlir::tts::GPU
