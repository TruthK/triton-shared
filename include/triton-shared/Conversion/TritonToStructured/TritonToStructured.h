#ifndef TRITON_TTS_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H
#define TRITON_TTS_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToStructuredPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_TTS_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H
