#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Conversion/LinalgToLLVM/Passes.h"
#include "triton-shared/Utils/PassUtils.h"

namespace mlir::tts {
struct LinalgToLLVMOptions : public PassPipelineOptions<LinalgToLLVMOptions> {
  PassOptions::Option<bool> reassociateFPReductions{
      *this, "reassociate-fp-reductions",
      llvm::cl::desc("Allow reassociation og FP reductions"),
      llvm::cl::init(false)};
};

using FunctionLikeNest = MultiOpNest<func::FuncOp>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void addCleanupPatterns(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);
}

void buildLinalgToLLVM(OpPassManager &pm, const LinalgToLLVMOptions &options) {
  pm.addPass(createConvertLinalgToAffineLoopsPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(bufferization::createOneShotBufferizePass());
  addCleanupPatterns(pm);

  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertLinalgToLoopsPass());
  // Expand complicated MemRef operations before lowering them.
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertSCFToCFPass());
  addCleanupPatterns(pm);

  pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createConvertComplexToLLVMPass());
  addCleanupPatterns(pm);

  pm.addPass(createConvertVectorToLLVMPass(
      // TODO: add more options on a per-need basis.
      ConvertVectorToLLVMPassOptions{options.reassociateFPReductions}));
  pm.addPass(createConvertIndexToLLVMPass());


  pm.addPass(memref::createExpandOpsPass());
  // Convert MemRef to LLVM (always needed).
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  addCleanupPatterns(pm);

  // Convert Func to LLVM (always needed).
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createLowerAffinePass());

  addCleanupPatterns(pm);

  pm.addPass(createArithToLLVMConversionPass());
  // Convert remaining unrealized_casts (always needed).
  pm.addPass(createReconcileUnrealizedCastsPass());
  addCleanupPatterns(pm);
}
} // namespace mlir::tts

namespace mlir {
namespace tts {
void registerLinalgToLLVM() {
  PassPipelineRegistration<LinalgToLLVMOptions>(
      "linalg-to-llvm",
      "An  pipeline to lower the main dialects (arith, linalg, "
      "memref, scf, vector) down to LLVM.",
      buildLinalgToLLVM);
}
} // namespace tts
} // namespace mlir
