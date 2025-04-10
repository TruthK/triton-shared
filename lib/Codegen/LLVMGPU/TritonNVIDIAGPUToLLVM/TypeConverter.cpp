#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/TypeConverter.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/Utility.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
using namespace mlir;
using namespace mlir::tts;

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &options,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, options, analysis) {
  // TODO
  // addConversion([ctx](triton::PointerType type) -> std::optional<Type> {
  //   assert(false);
  //   return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
  // });
}
