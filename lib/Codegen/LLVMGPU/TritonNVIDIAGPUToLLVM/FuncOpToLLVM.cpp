#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/TypeConverter.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::tts;

namespace {

inline Type u1Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 1, IntegerType::Unsigned);
}
// NOTE: [Additional Function Arguments]
// To support use of shared memory and global scratch memory inside of a
// function, the caller allocates a single large block of the relevant memory
// and calls the function with these extra arguments at the end.
// Specifically, the last argument is the global scratch memory allocation and
// the second to last is the shared memory allocation.
//
// For the kernel function itself, the shared memory base is a global symbol
// so no additional function argument is required but global scratch memory
// allocation is still passed in as the last argument. Though here the scratch
// memory is shared between all programs, so a linear offset based on the
// program id is required to get the local scratch base.

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public ConvertOpToLLVMPattern<func::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern(converter), numWarps(numWarps),
        targetInfo(targetInfo) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(func::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  func::FuncOp amendFuncOp(func::FuncOp funcOp,
                           ConversionPatternRewriter &rewriter,
                           const TargetInfoBase &targetInfo) const {
    // Push back two new arguments that indicate the current pointer to shared
    // memory and global scratch memory.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto sharedPtrTy =
        LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
    auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);

    // 1. Modify the function type to add the new arguments.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    bool isKernel = LLVM::tts::NVIDIA::isKernel(funcOp);
    if (!isKernel) {
      amendedInputTy.push_back(sharedPtrTy);
    }
    amendedInputTy.push_back(globalPtrTy);
    auto amendedFuncTy =
        FunctionType::get(ctx, amendedInputTy, funcTy.getResults());
    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    if (auto argAttrs = funcOp.getAllArgAttrs()) {
      llvm::SmallVector<mlir::Attribute> amendedArgAttrs(argAttrs.begin(),
                                                         argAttrs.end());
      while (amendedArgAttrs.size() < amendedInputTy.size()) {
        amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
      }
      amendedAttrs.push_back(
          rewriter.getNamedAttr(funcOp.getArgAttrsAttrName(),
                                rewriter.getArrayAttr(amendedArgAttrs)));
    }

    // 3. Add the new arguments to the region
    auto amendedFuncOp = rewriter.create<func::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
    if (!isKernel) {
      region.addArgument(sharedPtrTy, loc);
    }
    region.addArgument(globalPtrTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  // Map the MLIR attribute `tt.nv_tma_desc` to the appropriate LLVM and NVVM
  // attributes.
  static void handleByvalTmaDescArgs(LLVM::LLVMFuncOp &llvmFuncOp) {
    const bool isKernel = LLVM::tts::NVIDIA::isKernel(llvmFuncOp);
    for (unsigned i = 0; i < llvmFuncOp.getNumArguments(); ++i) {
      const auto attrs = llvmFuncOp.getArgAttrDict(i);
      if (!attrs) {
        continue;
      }
    }
  }

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = amendFuncOp(funcOp, rewriter, targetInfo);
    amendedFuncOp->dump();
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      funcOp->dump();
      return failure();
    }

    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;

    auto ctx = funcOp->getContext();

    if (LLVM::tts::NVIDIA::isKernel(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      newFuncOp->setAttr("nvvm.kernel", rewriter.getIntegerAttr(u1Ty(ctx), 1));
      newFuncOp.setLinkage(LLVM::Linkage::External);
    } else {
      // The noinline attribute will be used by the LLVM codegen to prevent
      // inlining.
      // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/LLVMIR/IR/LLVMInlining.cpp#L267
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
    }
    // Set an attribute for reqntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("nvvm.reqntid",
                       rewriter.getDenseI32ArrayAttr(32 * numWarps));

    rewriter.eraseOp(funcOp);
    rewriter.eraseOp(amendedFuncOp);

    // Add attributes for by-value TMA descriptor args (nvidia)
    handleByvalTmaDescArgs(newFuncOp);

    return success();
  }

private:
  int numWarps{0};
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::tts::NVIDIA::populateTTSFuncOpConversionPattern(
    RewritePatternSet &patterns, int numWarps,
    const TargetInfoBase &targetInfo) {
  mlir::LowerToLLVMOptions option(patterns.getContext());
  mlir::tts::MemrefToLLVMTypeConverter typeConverter(patterns.getContext(),
                                                     option);

  patterns.add<FuncOpConversion>(typeConverter, numWarps, targetInfo);
}
