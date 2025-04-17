// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== TileAndDistributeToWorkgroupsPass.cpp - Tile to workgroups pass ----===//
//
// This pass distributes the operations within the module to workgroups. This
// pass is created to move tile and distribution out of flow level and into
// the backends. For now this is mostly a bridge pass to connect things during
// the transition, and eventually might just be deprecated in favor of a
// utility method.
//
//===---------------------------------------------------------------------===//

#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Common/Transforms.h"
#include "triton-shared/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "triton-shared/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "triton-shared/Codegen/Transforms/Transforms.h"
#include "triton-shared/Codegen/Utils/Utils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-tile-and-distribute-to-workgroups"

namespace mlir::tts {

#define GEN_PASS_DEF_TILEANDDISTRIBUTETOWORKGROUPSPASS
#include "triton-shared/Codegen/Common/Passes.h.inc"

/// Method to return the configuration to use for first-level tile and
/// distribute. Returns the
/// - Root op of the dispatch. If no root op was found returns `nullptr`.
/// - tileSizes to use
/// - interchange
/// - loops to be partitioned (the tile sizes for the non-partitioned loop are
///   set to 0)
/// - static loop ranges - this is meant to be an optimization hint. It recovers
///   the static values that the workload of the dispatch corresponds to.
// TODO: Remove the use of static loop ranges. This is used to set the number of
// workgroups to a static value. Ideally this should not be done and the static
// and dyamic cases are handled the same way. When the tile+distribute moves
// away from using `scf.for` to using a construct that better captures
// distribution (like `scf.forall`) this information can be dropped.
static LogicalResult
getTileAndDistributeConfig(ArrayRef<Operation *> computeOps,
                           Operation *&dispatchRootOp,
                           SmallVectorImpl<int64_t> &tileSizes,
                           SmallVectorImpl<int64_t> &staticLoopRanges,
                           SmallVectorImpl<int64_t> &interchange,
                           SmallVectorImpl<unsigned> &partitionableLoops) {
  // Find the lowering configuration of the root operation.
  Operation *rootOp = nullptr;
  for (Operation *op : llvm::reverse(computeOps)) {
    if (getLoweringConfig(op)) {
      rootOp = op;
      break;
    }
  }
  if (!rootOp) {
    // THere is no lowering configuration. Return `null`.
    dispatchRootOp = nullptr;
    return success();
  }
  dispatchRootOp = rootOp;

  auto partitionableLoopInterface =
      dyn_cast<PartitionableLoopsInterface>(rootOp);
  if (!partitionableLoopInterface) {
    // Just return. All the in-out vectors are empty that should default
    // the number of workgroups to {1, 1, 1}
    return success();
  }

  partitionableLoops =
      partitionableLoopInterface.getPartitionableLoops(std::nullopt);
  IREE::Codegen::LoweringConfigAttrInterface rootOpConfig =
      getLoweringConfig(rootOp);
  if (!rootOpConfig) {
    return rootOp->emitOpError(
        "unable to find configuration of root op to define workgroup count "
        "region");
  }
  tileSizes.assign(rootOpConfig.getWorkgroupTileSizes());
  interchange.assign(rootOpConfig.getWorkgroupInterchange());

  // Set tile sizes of non-partitioned loops to 0.
  llvm::SmallDenseSet<unsigned> partitionableLoopsSet;
  partitionableLoopsSet.insert(partitionableLoops.begin(),
                               partitionableLoops.end());
  for (auto loopId : llvm::seq<unsigned>(0, tileSizes.size())) {
    if (partitionableLoopsSet.count(loopId))
      continue;
    tileSizes[loopId] = 0;
  }

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(*rootOp)) {
    staticLoopRanges = linalgOp.getStaticLoopRanges();
  }
  staticLoopRanges.resize(tileSizes.size(), ShapedType::kDynamic);

  return success();
}

//===---------------------------------------------------------------------===//
// Patterns to lower operations that are used to compute the number of
// workgroups.
//===---------------------------------------------------------------------===//


//===---------------------------------------------------------------------===//
// Patterns and methods for tile and distribute of Linalg ops to workgroups.
//===---------------------------------------------------------------------===//

namespace {

struct TileAndDistributeToWorkgroupsPass final
    : impl::TileAndDistributeToWorkgroupsPassBase<
          TileAndDistributeToWorkgroupsPass> {
  using impl::TileAndDistributeToWorkgroupsPassBase<
      TileAndDistributeToWorkgroupsPass>::TileAndDistributeToWorkgroupsPassBase;

  TileAndDistributeToWorkgroupsPass(
      int32_t maxWorkgroupParallelDims,
      linalg::DistributionMethod distributionMethod) {
    this->maxWorkgroupParallelDims = maxWorkgroupParallelDims;
    this->distributionMethod = distributionMethod;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect,  linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void TileAndDistributeToWorkgroupsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  auto funcOp = getOperation();

  // {
  //   RewritePatternSet patterns(context);
  //   populateReshapeToInterfaceTensorPatterns(patterns);
  //   if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
  //     funcOp.emitOpError("reshape to interface tensor patterns failed");
  //     return signalPassFailure();
  //   }
  // }

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  SmallVector<int64_t> tileSizes, staticLoopRanges, interchange;
  SmallVector<unsigned> partitionableLoops;
  Operation *dispatchRootOp = nullptr;
  if (failed(getTileAndDistributeConfig(computeOps, dispatchRootOp, tileSizes,
                                        staticLoopRanges, interchange,
                                        partitionableLoops))) {
    funcOp.emitOpError("failed to get tile and distribute configuration");
    return signalPassFailure();
  }

  IRRewriter rewriter(context);

  // If there are no compute ops, nothing more to do.
  if (!dispatchRootOp || computeOps.empty()) {
    return;
  }

  // Configure the linalg options.
  // Tile size selection function.
  auto tileSizeFn = [&](OpBuilder &builder,
                        Operation *op) -> SmallVector<Value> {
    // Check if tile sizes are deduced from the configuration. If so use
    // those.
    return llvm::map_to_vector(tileSizes, [&](int64_t ts) -> Value {
      return builder.create<arith::ConstantIndexOp>(op->getLoc(), ts);
    });
  };

  linalg::DistributionMethod distributionMethodValue =
      (linalg::DistributionMethod)(distributionMethod.getValue());
  auto linalgTilingOptions =
      linalg::LinalgTilingOptions()
          .setDistributionOptions(getIREELinalgLoopDistributionOptions(
              distributionMethodValue, maxWorkgroupParallelDims))
          .setInterchange(llvm::map_to_vector(
              interchange,
              [](int64_t v) -> unsigned { return static_cast<unsigned>(v); }))
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(tileSizeFn);

  FailureOr<IREETileAndFuseResult> tileAndFuseResult =
      tileAndFuseDispatchUsingSCFForOp(rewriter,
                                       cast<TilingInterface>(computeOps.back()),
                                       linalgTilingOptions);
  if (failed(tileAndFuseResult)) {
    funcOp.emitOpError("Tile+Distribute failed");
    return signalPassFailure();
  }

  // // Materialize the computation for workgroup counts.
  // if (exportOp) {
  //   auto workgroupCountOfr =
  //       getAsOpFoldResult(tileAndFuseResult->workgroupCount);

  //   // Resolve the `tensor.dim` operations in workgroup count region.
  //   {
  //     RewritePatternSet patterns(exportOp->getContext());
  //     memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  //     if (failed(
  //             applyPatternsGreedily(exportOp.value(), std::move(patterns)))) {
  //       exportOp->emitOpError("`tensor.dim` resolution in exportOp failed");
  //       return signalPassFailure();
  //     }
  //   }
  // }

  {
    RewritePatternSet patterns(context);
    populateTileAndDistributeToWorkgroupsCleanupPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("Tile+Distribute clean up patterns failed");
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Tile + Distribute ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  {
    SmallVector<int64_t> staticNumWorkgroup = getStaticNumWorkgroups(funcOp);
    // Apply linalg tiling optimization patterns, which includes folding
    // casting ops into tiled operations.
    RewritePatternSet patterns(context);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    populateFoldAffineMinInDistributedLoopsPatterns(patterns,
                                                    staticNumWorkgroup);
    context->getOrLoadDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("tiling canonicalizations failed");
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Canonicalize ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // After rewriting destructive updates, there might be uses of compute
  // operations only in `tensor.dim` ops. Resolve these.
  RewritePatternSet resolveDimOps(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(resolveDimOps);
  if (failed(applyPatternsGreedily(funcOp, std::move(resolveDimOps)))) {
    funcOp.emitOpError("resolving ranked shaped results dims failed");
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileAndDistributeToWorkgroupsPass(
    int32_t maxWorkgroupParallelDims,
    linalg::DistributionMethod distributionMethod) {
  return std::make_unique<TileAndDistributeToWorkgroupsPass>(
      maxWorkgroupParallelDims, distributionMethod);
}

} // namespace mlir::iree_compiler
