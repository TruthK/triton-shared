#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/TypeConverter.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/Utility.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
using namespace mlir;
using namespace mlir::tts;

MemrefToLLVMTypeConverter::MemrefToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &options,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, options, analysis) {
  // TODO  no work
  addConversion([ctx](mlir::UnrankedMemRefType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, 1);
  });
}
