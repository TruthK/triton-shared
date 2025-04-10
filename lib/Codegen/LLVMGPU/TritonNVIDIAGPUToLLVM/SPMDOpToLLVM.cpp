#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::tts;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetNumProgramsOp. If numCTAs = 1, then
    // GetNumProgramsOp is converted to "%nctaid", otherwise it is converted to
    // "%nclusterid".
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for GetProgramIdOp");
    // TODO triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);
    assert(false);
    int numCTAs = 1;

    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    std::string sreg = numCTAs == 1 ? "nctaid." : "nclusterid.";
    sreg.append(1, 'x' + op.getAxisAsInt()); // 0 -> 'x', 1 -> 'y', 2 -> 'z'

    Value numPrograms = LLVM::tts::NVIDIA::getSRegValue(rewriter, loc, sreg);
    rewriter.replaceOp(op, numPrograms);
    return success();
  }
};


struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  explicit GetProgramIdOpConversion(LLVMTypeConverter &typeConverter,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::GetProgramIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value programId = targetInfo.programId(rewriter, op->getLoc(),
                                           op->getParentOfType<ModuleOp>(),
                                           op.getAxisAsInt());
    rewriter.replaceOp(op, programId);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::tts::NVIDIA::populateTTSSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter,
    const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<GetProgramIdOpConversion>(typeConverter, targetInfo, benefit);
}
