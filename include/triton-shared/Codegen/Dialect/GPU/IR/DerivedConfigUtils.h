#ifndef TRITON_SHARED_CODEGEN_DIALECT_GPU_IR_DERIVEDCONFIGUTILS_H_
#define TRITON_SHARED_CODEGEN_DIALECT_GPU_IR_DERIVEDCONFIGUTILS_H_

#include "mlir/IR/Operation.h"

namespace mlir::tts::GPU {

SmallVector<int64_t> deriveThreadTileSizes(Operation *op);

} // namespace mlir::tts::GPU

#endif // TRITON_SHARED_CODEGEN_DIALECT_GPU_IR_DERIVEDCONFIGUTILS_H_