#include "triton-shared/Codegen/Common/GPU/GPUPatterns.h"
#include "triton-shared/Codegen/Common/Transforms.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "triton-shared/Codegen/LLVMGPU/ConvertToLLVM.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/TypeConverter.h"
#include "triton-shared/Codegen/Utils/GPUUtils.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-convert-to-nvvm"

namespace mlir::tts {

#define GEN_PASS_DEF_CONVERTTONVVMPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct ConvertToNVVMPass final
    : impl::ConvertToNVVMPassBase<ConvertToNVVMPass> {
  using impl::ConvertToNVVMPassBase<ConvertToNVVMPass>::ConvertToNVVMPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<gpu::GPUDialect, IREE::GPU::IREEGPUDialect, LLVM::LLVMDialect,
                NVVM::NVVMDialect, affine::AffineDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(m.getContext(), DataLayout(m));
    options.overrideIndexBitwidth(64);
    LLVMTypeConverter converter(m.getContext(), options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kGlobalMemorySpace);
          case gpu::AddressSpace::Workgroup:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kSharedMemorySpace);
          case gpu::AddressSpace::Private:
            return 0;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
    // Lowering for MMAMatrixType.
    converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToLLVMType(type);
    });
    // Convert dummy tokens.
    converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) -> Type {
      return converter.convertType(IntegerType::get(type.getContext(), 32));
    });

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    // Run Vector -> Vector transformations ahead of conversion to LLVM.
    {
      RewritePatternSet patterns(&getContext());
      populateVectorToSCFConversionPatterns(
          patterns, VectorTransferToSCFOptions().enableFullUnroll());
      populateDropSharedMemoryDeallocOpPatterns(patterns);
      populateScalarizeMathOps(patterns);
      populateConvertSharedMemoryAllocOps(patterns);
      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      // We currently always use 64 bit indices, thus ensure the bit width of
      // the mask compare is consistent.
      vector::populateVectorMaskMaterializationPatterns(
          patterns, /*force32BitVectorIndices=*/false);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      // TODO: doubtful that the "default" does what one want here, it is likely
      // better to use something else.
      vector::populateVectorTransposeLoweringPatterns(
          patterns, vector::VectorTransformsOptions());
      vector::populateVectorTransferLoweringPatterns(patterns);
      arith::populateExpandBFloat16Patterns(patterns);
      if (failed(applyPatternsGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      populateGpuRewritePatterns(patterns);
      if (failed(applyPatternsGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    {
      // Convert arith::maximumf/minimumf ops on older gpus since the lowering
      // is faulty for them.
      // TODO: Remove this once the lowering in LLVM is fixed
      // (https://github.com/llvm/llvm-project/issues/64606).
      auto attr = getGPUTargetAttr(*(m.getOps<FunctionOpInterface>().begin()));
      std::optional<int> cc = attr.getCUDAComputeCapability();
      if (!cc || cc.value() < 80) {
        RewritePatternSet patterns(&getContext());
        populateReplaceSlowMinMaxOpsPatterns(patterns);
        if (failed(applyPatternsGreedily(m, std::move(patterns)))) {
          return signalPassFailure();
        }
      }
    }
    {
      mlir::tts::NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
      RewritePatternSet llvmPatterns(&getContext());
      // populateLowerHALInterfaceOp(llvmPatterns);
      // populateLLVMConversionPatterns(&getContext(), llvmPatterns, converter);
      populateComplexToLLVMConversionPatterns(converter, llvmPatterns);
      populateMathToLLVMConversionPatterns(converter, llvmPatterns);
      memref::populateExpandStridedMetadataPatterns(llvmPatterns);
      populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
      mlir::tts::MemrefToLLVMTypeConverter typeConverter(m.getContext(),
                                                         options);
      mlir::tts::NVIDIA::populateTTSFuncOpConversionPattern(
          typeConverter, llvmPatterns, 4, targetInfo);
      cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
      arith::populateCeilFloorDivExpandOpsPatterns(llvmPatterns);
      arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
      vector::populateVectorRankReducingFMAPattern(llvmPatterns);
      vector::populateVectorInsertExtractStridedSliceTransforms(llvmPatterns);
      vector::populateVectorStepLoweringPatterns(llvmPatterns);
      populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
      vector::populateVectorTransferLoweringPatterns(llvmPatterns,
                                                     /*maxTransferRank=*/1);
      populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
      populateNVGPUToNVVMConversionPatterns(converter, llvmPatterns);
      populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);

      /// Target specification.
      LLVMConversionTarget target(getContext());
      target.addIllegalOp<func::FuncOp>();
      target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
      target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
      target.addIllegalDialect<gpu::GPUDialect>();
      target.addIllegalOp<
          LLVM::CopySignOp, LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op,
          LLVM::FAbsOp, LLVM::FCeilOp, LLVM::FFloorOp, LLVM::FRemOp,
          LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op, LLVM::PowOp,
          LLVM::RoundEvenOp, LLVM::RoundOp, LLVM::SinOp, LLVM::SqrtOp>();

      // TODO: Remove once we support replacing non-root ops.
      target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp>();

      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns)))) {
        signalPassFailure();
      }
    }
    // Convert NVVM ops to Inline Assembly.
    {
      RewritePatternSet patterns(&getContext());
      populateNVVMToLLVMConversionPatterns(patterns);
      if (failed(applyPatternsGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    ConvertToDynamicSharedMemory(m);
  }
};

} // namespace
} // namespace mlir::tts
