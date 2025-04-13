#ifndef TRITON_TTS_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_TTS_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/TargetInfoBase.h"

namespace mlir {
namespace tts {

namespace NVIDIA {

// void populateElementwiseOpToLLVMPatterns(
//     LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
//     ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
//     const TargetInfo &targetInfo, PatternBenefit benefit);

void populateTTSSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                    const TargetInfoBase &targetInfo,
                                    RewritePatternSet &patterns);

void populateTTSFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        int numWarps,
                                        const TargetInfoBase &targetInfo);

void populateTTSKernelArgCleanupPattern(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns);

// void populateClampFOpToLLVMPattern(LLVMTypeConverter &typeConverter,
//                                    RewritePatternSet &patterns,
//                                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
//                                    int computeCapability,
//                                    PatternBenefit benefit);

} // namespace NVIDIA
} // namespace tts
} // namespace mlir

#endif
