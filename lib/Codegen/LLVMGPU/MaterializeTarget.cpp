#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "triton-shared/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "triton-shared/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "triton-shared/Codegen/LLVMGPU/KernelConfig.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"

namespace mlir::tts {

#define GEN_PASS_DEF_MATERIALIZETARGETPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {
struct CUDAOptions {
  std::string clTarget = "sm_89";
  std::string clTargetFeatures = "+ptx84";
  // bool clUsePtxas = false;
  // std::string clUsePtxasFrom;
  // std::string clUsePtxasParams;

  LogicalResult verify(mlir::Builder &builder) const {
    if (IREE::GPU::normalizeCUDATarget(clTarget).empty()) {
      return emitError(builder.getUnknownLoc(), "Unknown CUDA target '")
             << clTarget << "'";
    }
    return success();
  }
};

/// Selects a lowering strategy for taking a hal.executable.variant operation
/// to scalar/native-vector code.
class MaterializeTargetPass final
    : public impl::MaterializeTargetPassBase<MaterializeTargetPass> {
public:
  using impl::MaterializeTargetPassBase<
      MaterializeTargetPass>::MaterializeTargetPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect, IREE::Codegen::IREECodegenDialect,
                    IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override;
};
} // namespace

IREE::GPU::ExecutableTargetAttr
getExecutableTarget(MLIRContext *context, const CUDAOptions &options) {
  Builder b(context);
  SmallVector<NamedAttribute> configItems;
  auto addConfig = [&](StringRef name, Attribute value) {
    configItems.emplace_back(b.getStringAttr(name), value);
  };

  if (failed(options.verify(b)))
    return nullptr;

  if (auto target = IREE::GPU::getCUDATargetDetails(
          options.clTarget, options.clTargetFeatures, context))
    addConfig("iree.gpu.target", target);

  return b.getAttr<IREE::GPU::ExecutableTargetAttr>(
      b.getStringAttr("cuda"), b.getStringAttr("cuda-nvptx-fb"),
      b.getDictionaryAttr(configItems));
}

void MaterializeTargetPass::runOnOperation() {
  auto moduleOp = getOperation();
  CUDAOptions options;
  options.clTarget = "sm_" + std::to_string(computeCapability.getValue());
  options.clTargetFeatures = "+ptx" + std::to_string(ptxVersion.getValue());

  // Create an IntegerAttr with the value
  IntegerAttr warpAttr =
      IntegerAttr::get(IntegerType::get(moduleOp.getContext(), 32), numWrap);

  moduleOp.walk([&](func::FuncOp funcOp) {
    // Add the attribute to the ModuleOp with a name "num_warp"
    funcOp->setAttr("num_warp", warpAttr);
    // funcOp->setAttr("llvm.bareptr", BoolAttr::get(funcOp.getContext(),
    // true));
  });

  auto targetsAttr = getExecutableTarget(&getContext(), options);
  if (!targetsAttr) {
    return signalPassFailure();
  }
  moduleOp->setAttr("hal.device.targets",
                    ArrayAttr::get(moduleOp.getContext(), targetsAttr));
}
} // namespace mlir::tts
