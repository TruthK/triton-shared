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
  // TODO
  addConversion([ctx](mlir::UnrankedMemRefType type) -> std::optional<Type> {
    type.dump();
     type.dump();
    return LLVM::LLVMPointerType::get(ctx, 1);
  });
}
