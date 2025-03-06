#ifndef TTS_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_CONFIGUTILS_H_
#define TTS_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_CONFIGUTILS_H_

#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenAttrs.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUAttrs.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::tts::GPU {

LogicalResult setDataTiledMultiMmaLoweringConfig(
    tts::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    Operation *op, tts::GPU::UKernelConfigAttr ukernelConfig);

LogicalResult
setIGEMMConvolutionLoweringConfig(tts::GPU::TargetAttr target,
                                  mlir::FunctionOpInterface entryPoint,
                                  Operation *op);

LogicalResult setMatmulLoweringConfig(tts::GPU::TargetAttr target,
                                      mlir::FunctionOpInterface entryPoint,
                                      Operation *op);

LogicalResult setTileAndFuseLoweringConfig(tts::GPU::TargetAttr target,
                                           mlir::FunctionOpInterface entryPoint,
                                           Operation *op);

struct GPUPipelineOptions {
  bool enableReduceSharedMemoryBankConflicts = true;
  bool prefetchSharedMemory = false;
  bool useIgemmConvolution = false;
  bool enableUkernels = false;
  std::optional<ReorderWorkgroupsStrategy> reorderStrategy;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GPUPipelineOptions &options);

GPUPipelineOptions
getPipelineOptions(FunctionOpInterface funcOp,
                   TTS::Codegen::TranslationInfoAttr translationInfo);

} // namespace mlir::tts::GPU

#endif // TTS_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_CONFIGUTILS_H_
