#ifndef TRITON_TTS_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_TTS_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/TargetInfoBase.h"

using namespace mlir;
namespace mlir::tts {
class MemrefToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  MemrefToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                          
                               const DataLayoutAnalysis *analysis = nullptr);


};
}
#endif
