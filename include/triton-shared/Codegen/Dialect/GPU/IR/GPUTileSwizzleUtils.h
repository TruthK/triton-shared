#ifndef TTS_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_
#define TTS_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_

#include "triton-shared/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUAttrs.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUEnums.h"

namespace mlir::tts::GPU {

Codegen::TileSwizzle getIntrinsicSwizzle(tts::GPU::MMAIntrinsic intrinsic,
                                         tts::GPU::MMAFragment fragment);

Codegen::TileSwizzle getSwizzle(tts::GPU::DataTiledMMAAttr mma,
                                tts::GPU::MMAFragment fragment);

} // namespace mlir::tts::GPU

#endif // TTS_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_
