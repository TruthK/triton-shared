#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;

namespace mlir::tts {

#define GEN_PASS_DEF_TILETTSTRANSFERWRITEPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Selects a lowering strategy for taking a hal.executable.variant operation
/// to scalar/native-vector code.
class TileTTSTransferWritePass final
    : public impl::TileTTSTransferWritePassBase<TileTTSTransferWritePass> {
public:
  using impl::TileTTSTransferWritePassBase<
      TileTTSTransferWritePass>::TileTTSTransferWritePassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                    mlir::tts::TritonStructuredDialect>();
    mlir::tts::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override;
};
} // namespace

void TileTTSTransferWritePass::runOnOperation() {
  MLIRContext *context = &getContext();

  // 获取函数操作
  auto funcOp = getOperation();

  // 遍历函数内部查找TTS TransferWriteOp操作
  funcOp.walk([&](tts::TransferWriteOp transferWriteOp) {
    // 获取TransferWriteOp的位置，用于放置新的操作
    IRRewriter rewriter(transferWriteOp);

    // 获取写入的值
    Value value = transferWriteOp.getValue();

    // 查找值的定义操作，检查是否是scf::ForallOp
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || !isa<scf::ForallOp>(definingOp)) {
      // 如果不是scf::ForallOp，则跳过此操作
      return;
    }

    // 获取scf::ForallOp
    auto forallOp = cast<scf::ForallOp>(definingOp);

    // 分析forallOp的迭代域，计算tile大小
    SmallVector<OpFoldResult> tileSizes;

    // 获取forall的边界和步长
    for (unsigned i = 0; i < forallOp.getRank(); i++) {
      // 获取上界和步长
      Value ub = forallOp.getUpperBound(rewriter)[i];
      Value lb = forallOp.getLowerBound(rewriter)[i];
      Value step = forallOp.getStep(rewriter)[i];

      // 如果是常量，我们可以计算实际迭代次数
      auto ubConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(ub.getDefiningOp());
      auto lbConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(lb.getDefiningOp());
      auto stepConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(step.getDefiningOp());
      if (ubConstant && lbConstant && stepConstant) {
        // 计算tile大小为forall的步长
        tileSizes.push_back(stepConstant.getValue());
      } else {
        // 如果不是常量，使用步长作为tile大小
        tileSizes.push_back(step);
      }
    }

    // 设置tiling选项
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    SmallVector<Attribute> mapping;
    int idx = 0;
    for (auto size : tileSizes) {
      if (!isConstantIntValue(size, 0)) {
        unsigned mappingId =
            static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
        mapping.push_back(gpu::GPUThreadMappingAttr::get(
            context, static_cast<gpu::MappingId>(mappingId)));
        // if (tilingLevel == IREE::GPU::TilingLevel::Thread) {
        //   mapping.push_back(gpu::GPUThreadMappingAttr::get(
        //       context, static_cast<gpu::MappingId>(mappingId)));
        // } else {
        //   // Else it must be subgroup tiling.
        //   mapping.push_back(gpu::GPUWarpMappingAttr::get(
        //       context, static_cast<gpu::MappingId>(mappingId)));
        // }
      }
    }
    tilingOptions.setMapping(llvm::to_vector(llvm::reverse(mapping)));

    // 使用TilingInterface进行tiling
    auto tilingInterfaceOp =
        cast<mlir::TilingInterface>(transferWriteOp.getOperation());
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, tilingOptions);

    // 检查tiling是否成功
    if (succeeded(tilingResult)) {
      // 如果成功，替换原始操作的结果（如果有）
      // transfer_write操作没有结果，所以只需删除原始操作
      transferWriteOp.erase();
    }

    // Apply cleanup patterns.
    {
      FunctionOpInterface funcOp = getOperation();
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      scf::ForOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitError() << "tiling cleanup failed\n";
        return signalPassFailure();
      }
    }
  });
}
} // namespace mlir::tts
