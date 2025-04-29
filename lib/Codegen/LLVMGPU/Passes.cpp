// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Codegen/Common/GPU/Passes.h"
#include "triton-shared/Codegen/Common/PassUtils.h"
#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "triton-shared/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Codegen/Utils/GPUUtils.h"
#include "triton-shared/Codegen/Utils/MarkerUtils.h"
#include "triton-shared/Codegen/Utils/Utils.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Utils/PassUtils.h"

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "iree-llvm-gpu-lowering-pass-pipeline"

namespace mlir::tts {

constexpr int64_t kDefaultSubgroupSize = 32;

static IREE::GPU::ReorderWorkgroupsStrategy clReorderWorkgroupsStrategy =
    IREE::GPU::ReorderWorkgroupsStrategy::None;

static int64_t clLLVMGPUSharedMemoryLimit = 163 * 1024;

static bool clLLVMGPUEnableSharedMemoryReuse = false;

//===----------------------------------------------------------------------===//
// Bufferization Configuration
//===----------------------------------------------------------------------===//

static bool hasThreadMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) {
    return false;
  }
  return llvm::any_of(*forall.getMapping(),
                      llvm::IsaPred<gpu::GPUThreadMappingAttr>);
}

// All pipelines that use this allocation function distribute scf.forall ops
// after bufferizing. This means that to differentiate between an allocation in
// function memory and workgroup memory, we need to look for a parent
// scf.forall op with a thread mapping. If not present, we allocate workgroup
// memory. Pipelines that choose to distribute in a different order will have
// to use a different allocation function.
static FailureOr<Value> gpuAllocationFn(OpBuilder &builder, Location loc,
                                        MemRefType memRefType,
                                        ValueRange dynamicSizes,
                                        unsigned alignment) {
  Block *insertionBlock = builder.getInsertionBlock();
  Operation *parent = insertionBlock->getParentOp();
  scf::ForallOp enclosingForall = dyn_cast<scf::ForallOp>(parent);
  if (!enclosingForall) {
    enclosingForall = parent->getParentOfType<scf::ForallOp>();
  }
  if (enclosingForall && hasThreadMapping(enclosingForall)) {
    auto addressSpace = gpu::AddressSpaceAttr::get(
        builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), addressSpace);
    return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes)
        .getResult();
  }

  auto addressSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), addressSpace);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

// Barriers are only needed when copying to/from workgroup memory. The only
// other kind of memory that can be allocated is function memory, which is local
// to a thread.
static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  bool needsBarrier = false;
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(from.getType()))) {
    needsBarrier = true;
  }
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier)
    builder.create<gpu::BarrierOp>(loc);
  Operation *copy = builder.create<memref::CopyOp>(loc, from, to);
  if (needsBarrier) {
    setMarker(copy, getCopyToWorkgroupMemoryMarker());
    builder.create<gpu::BarrierOp>(loc);
  }
  return success();
}

// Returns success when workgroup reordering is supported / enabled for
// `funcOp`. On ROCm, we require workgroup counts to be static.
static LogicalResult canReorderWorkgroups(FunctionOpInterface funcOp) {
  auto target = IREE::GPU::ExecutableTargetAttr::lookup(funcOp);
  if (!target) {
    return failure();
  }
  if (target.getBackend() != "rocm")
    return success();

  // Workgroup reordering on ROCm currently requires all workgrup counts to be
  // static.
  SmallVector<int64_t> workgroupCounts = getStaticNumWorkgroups(funcOp);
  if (llvm::any_of(workgroupCounts, ShapedType::isDynamic))
    return failure();

  // This is further restricted to 2D+ grids as we reorder along the X and Y
  // workgroup IDs.
  return success(workgroupCounts.size() >= 2);
}

// Reconciles workgroup reordering strategy based on the pipeline `option` and
// the CLI flag.
static IREE::GPU::ReorderWorkgroupsStrategy getReorderWorkgroupsStrategy(
    const std::optional<IREE::GPU::ReorderWorkgroupsStrategy> &option) {
  return option.value_or(clReorderWorkgroupsStrategy);
}

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

static void addBufferizePasses(OpPassManager &funcPassManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

static void tileAndDistributeToWorkgroup(
    OpPassManager &funcPassManager, bool useForall,
    std::optional<ConvertToDestinationPassingStylePassOptions>
        convertToDpsOptions = ConvertToDestinationPassingStylePassOptions{},
    ReorderWorkgroupsStrategy strategy = ReorderWorkgroupsStrategy::None) {
  if (useForall) {
    // funcPassManager.addPass(
    // createTileAndDistributeToWorkgroupsUsingForallOpPass());
    // funcPassManager.addPass(createTransferReadSubviewFusionPass());
  } else {
    funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass(
        kNumMaxParallelDims,
        linalg::DistributionMethod::CyclicNumProcsEqNumIters));
    funcPassManager.addPass(createCSEPass());
    if (convertToDpsOptions) {
      funcPassManager.addPass(
          createConvertToDestinationPassingStylePass(*convertToDpsOptions));
    }
  }
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

static void tileAndBufferize(OpPassManager &funcPassManager) {
  ConvertToDestinationPassingStylePassOptions options;
  options.useWARForCooperativeMatrixCodegen = true;
  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true, options);
  addBufferizePasses(funcPassManager);
}

static void addGPUVectorizationPasses(OpPassManager &funcPassManager,
                                      bool vectorizeCopies = true) {
  funcPassManager.addPass(
      IREE::VectorExt::createVectorizeIREEVectorExtOpsPass());
  // Vectorize.
  GenericVectorizationPassOptions options;
  options.vectorizePadding = true;
  options.vectorizeCopies = vectorizeCopies;
  options.vectorizeGatherAccesses = true;
  options.enableCleanup = false;
  options.foldCastIntoContract = true;
  funcPassManager.addPass(createGenericVectorizationPass(options));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  // Run subset hoisting to convert iter_args to vectors.
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

// //===---------------------------------------------------------------------===//
// // Default Vectorization
// //===---------------------------------------------------------------------===//

// void addGPUVectorizationPassPipeline(OpPassManager &funcPassManager) {
//   tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/false);

//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   // Distribute linalg onto threads within the workgroup.
//   funcPassManager.addPass(createGPUTensorTilePass());
//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   // Linalg -> vector
//   addGPUVectorizationPasses(funcPassManager);

//   // tensor to memref
//   addBufferizePasses(funcPassManager);
//   funcPassManager.addPass(createGPUDistributePass());

//   // Post bufferization optimizations.
//   funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
//   funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
//   funcPassManager.addPass(createOptimizeVectorTransferPass());
//   funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
// }

//===---------------------------------------------------------------------===//
// Tile and Fuse
//===---------------------------------------------------------------------===//

static FailureOr<Value> gpuRequireMemSpaceAllocationFn(OpBuilder &builder,
                                                       Location loc,
                                                       MemRefType memRefType,
                                                       ValueRange dynamicSizes,
                                                       unsigned alignment) {
  Attribute memorySpace = memRefType.getMemorySpace();
  // Bail out if the memref type specifies a nonnull memory space that is not
  // #gpu.address_space.
  if (memorySpace && !llvm::isa<gpu::AddressSpaceAttr>(memorySpace)) {
    return failure();
  }

  MemRefType allocType = memRefType;
  auto privateSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  if (!memorySpace) {
    allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), privateSpace);
    memorySpace = privateSpace;
  }

  if (memorySpace == privateSpace) {
    return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes)
        .getResult();
  }
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

static void addGPUBufferizePasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createEliminateEmptyTensorsPass());
  funcPassManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  funcPassManager.addPass(createGPUInferMemorySpacePass());
  BufferizationOptions::AllocationFn allocationFn =
      gpuRequireMemSpaceAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = [](OpBuilder &builder, Location loc,
                                               Value from, Value to) {
    builder.create<memref::CopyOp>(loc, from, to);
    return success();
  };

  funcPassManager.addPass(triton::createConvertTTSTransferOpPass());
  funcPassManager.addPass(
      createIREEComprehensiveBufferizePass(allocationFn, memcpyFn));
  addIREEPostBufferizationPasses(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

/// Control function for decomposing pack and unpack ops. Returns true ifthe
/// op is a PackOp with a DispatchTensorLoadOp producer, or an UnPackOpwith
/// only DispatchTensorStoreOp consumers.
// LogicalResult isAtBoundary(Operation *op) {
//   assert(false && "isAtBoundary is not implemented");
//   if (isa<tensor::PackOp>(op)) {
//     // if (isa_and_nonnull<IREE::Flow::DispatchTensorLoadOp>(
//     //         op->getOperand(0).getDefiningOp())) {
//     //   return success();
//     // }
//   } else if (isa<tensor::UnPackOp>(op)) {
//     // if (llvm::all_of(op->getUsers(), [](Operation *user) {
//     //       return isa<IREE::Flow::DispatchTensorStoreOp>(user);
//     //     })) {
//     //   return success();
//     // }
//   }
//   return failure();
// }

void addGPUTileAndFusePassPipeline(OpPassManager &funcPassManager,
                                   const GPUPipelineOptions &pipelineOptions) {

  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true,
                               std::nullopt);

  // Step 1. Promote matmul operands and pack to intrinsic shapes.
  // funcPassManager.addPass(createGPUPadOperandsPass());
  //   funcPassManager.addPass(createGPUPromoteMatmulOperandsPass());
  //   funcPassManager.addPass(createGPUPackToIntrinsicsPass());
  //   // Decompose packs and unpacks that are at the function boundary.
  //   funcPassManager.addPass(createDecomposeBoundaryPackUnPackOpsPass());

  // funcPassManager.addPass(createPropagateReshapesByExpansionPass());

  // Step 2. Tile and fuse tileable ops to reduction loops.
  // {
  //   GPUApplyTilingLevelPassOptions options;
  //   options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
  //   funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
  //   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  //   funcPassManager.addPass(createCSEPass());
  // }

  // funcPassManager.addPass(createPropagateReshapesByExpansionPass());
  // funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  // funcPassManager.addPass(createCSEPass());

  // Step 4. Tile and fuse tileable ops to subgroups/threads.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Thread;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(createTileTTSTransferWritePass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Subgroup;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
  }

  // Step 4.5. Things that need to happen right after distribution to
  // threads. funcPassManager.addPass(createGPULowerToUKernelsPass());

  // Normalize loop bounds for later lowerings.
  funcPassManager.addPass(mlir::tts::createNormalizeLoopBoundsPass(
      NormalizeLoopBoundsPassOptions{/*normalizeFor=*/false,
                                     /*normalizeForall=*/true}));
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Step 5. Greedily fuse parallel loops and hoist from serial loops.
  funcPassManager.addPass(createGPUFuseAndHoistParallelLoopsPass());
  funcPassManager.addPass(createGPUGreedilyDistributeToThreadsPass());
  funcPassManager.addPass(createTileLargeTensorsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(IREE::GPU::createCombineBarrierRegionsPass());

  // Step 6. Lower special ops and vectorize.
  funcPassManager.addPass(IREE::GPU::createVectorizeIREEGPUOpsPass());
  addGPUVectorizationPasses(funcPassManager, /*vectorizeCopies=*/false);
  funcPassManager.addPass(createCleanupBufferAllocViewPass());
  funcPassManager.addPass(createGPUCombineValueBarriersPass());

  //   // Step 7. Bufferize.
  addGPUBufferizePasses(funcPassManager);
  funcPassManager.addPass(createSPMDOpPass());
  // Step 8. Resolve remaining parallel loops.
  funcPassManager.addPass(mlir::tts::createNormalizeLoopBoundsPass(
      NormalizeLoopBoundsPassOptions{/*normalizeFor=*/false,
                                     /*normalizeForall=*/true}));
  funcPassManager.addPass(createGPUVerifyDistributionPass());
  funcPassManager.addPass(createGPUDistributeForallPass());

  // Vectorize copies that came out of bufferization.
  funcPassManager.addPass(createVectorExtTransferToVectorTransferPass());
  funcPassManager.addPass(createTransferOpCanonicalizePass());
  funcPassManager.addPass(createVectorizeMemrefCopyPass());

  // Step 8. Unroll operations to native intrinsic widths.
  funcPassManager.addPass(IREE::GPU::createUnrollToIntrinsicsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Step 9. Remaining post-bufferization optimizations/lowerings.
  funcPassManager.addPass(IREE::GPU::createLowerIREEGPUOpsPass());
  funcPassManager.addPass(createUnrollAnnotatedLoopsPass());
  // funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  if (pipelineOptions.enableReduceSharedMemoryBankConflicts) {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 64;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }
  if (pipelineOptions.prefetchSharedMemory) {
    funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
    funcPassManager.addPass(createLLVMGPUPrefetchSharedMemoryPass());
  }

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  {
    OptimizeVectorTransferPassOptions options;
    // Disable redundant vector transfer hoisting because it does not
    // properly consider distributed code on memrefs.
    options.redundantHoisting = false;
    funcPassManager.addPass(createOptimizeVectorTransferPass());
  }
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

// //===---------------------------------------------------------------------===//
// // Winograd Vectorize
// //===---------------------------------------------------------------------===//

// // void addGPUWinogradVectorizePassPipeline(OpPassManager &funcPassManager)
// {
// //   tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true);

// //   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
// //   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
// //   funcPassManager.addPass(createCSEPass());

// //   // Distribute linalg onto threads within the workgroup.
// //   funcPassManager.addPass(createGPUTilePass());
// //   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
// //   funcPassManager.addPass(createCSEPass());
// //   funcPassManager.addPass(
// //       IREE::LinalgExt::createDecomposeWinogradTransformPass());

// //   // Linalg -> vector
// //   addGPUVectorizationPasses(funcPassManager);

// //   // tensor to memref
// //   addBufferizePasses(funcPassManager);
// //   GPUDistributeScfForPassOptions options;
// //   options.useBlockDims = false;
// //   funcPassManager.addPass(createGPUDistributeScfForPass(options));

// //   // Post bufferization optimizations.
// //   funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
// //   funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
// //   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
// //   funcPassManager.addPass(createCSEPass());
// //   funcPassManager.addPass(createOptimizeVectorTransferPass());
// // funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
// // }

// //===---------------------------------------------------------------------===//
// // Matmul Tensor Core
// //===---------------------------------------------------------------------===//

// void addGPUMatmulTensorCorePassPipeline(OpPassManager &funcPassManager,
//                                         const GPUPipelineOptions &options,
//                                         unsigned pipelineDepth) {
//   tileAndBufferize(funcPassManager);

//   // Distribute linalg onto warps within the workgroup.
//   funcPassManager.addPass(
//       createLLVMGPUTileAndDistributePass(/*distributeToWarp=*/true));
//   funcPassManager.addPass(createRemoveSingleIterationLoopPass());
//   if (pipelineDepth > 1) {
//     funcPassManager.addPass(createGPUMultiBufferingPass(
//         GPUMultiBufferingPassOptions{pipelineDepth}));
//   }
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   funcPassManager.addPass(createRemoveSingleIterationLoopPass());

//   IREE::GPU::ReorderWorkgroupsStrategy reorderStrategy =
//       getReorderWorkgroupsStrategy(options.reorderStrategy);
//   funcPassManager.addPass(
//       createReorderWorkgroups(reorderStrategy, canReorderWorkgroups));

//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   // Linalg -> vector
//   funcPassManager.addPass(
//       createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType::WMMA));
//   funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
//   funcPassManager.addPass(createCSEPass());
//   funcPassManager.addPass(createOptimizeVectorTransferPass());
//   funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

//   // Distribute shared memory copies.
//   funcPassManager.addPass(createMemrefCopyToLinalgPass());
//   funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
//   if (options.enableReduceSharedMemoryBankConflicts) {
//     funcPassManager.addPass(createGPUReduceBankConflictsPass());
//   }

//   // Vector -> MMA ops
//   funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
//   funcPassManager.addPass(
//       createLLVMGPUVectorToGPUPass(GPUTensorCoreType::WMMA));
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   // Hoist loop invariant code to avoid pipelining it.
//   funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
//   // Pipeline memory operations.
//   GPUPipeliningPassOptions pipelieningOptions = {};
//   pipelieningOptions.epiloguePeeling = false;
//   pipelieningOptions.depth = pipelineDepth;
//   pipelieningOptions.scheduleIndex =
//       llvm::to_underlying(PipeliningSchedulingStrategy::loadGlobalStage0);
//   funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));
//   // Optimize shared memory usage.
//   funcPassManager.addPass(createLLVMGPUPackSharedMemoryAllocPass());
// }

//===---------------------------------------------------------------------===//
// Matmul MMA.Sync
//===---------------------------------------------------------------------===//

void addGPUMatmulTensorCoreMmaSyncPassPipeline(
    OpPassManager &funcPassManager, const GPUPipelineOptions &options,
    unsigned pipelineDepth) {
  tileAndBufferize(funcPassManager);
  funcPassManager.addPass(createSPMDOpPass());
  // Distribute linalg onto warps within the workgroup.
  funcPassManager.addPass(
      createLLVMGPUTileAndDistributePass(/*distributeToWarp=*/true));
  // funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  if (pipelineDepth > 1) {
    funcPassManager.addPass(createGPUMultiBufferingPass(
        GPUMultiBufferingPassOptions{pipelineDepth}));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  IREE::GPU::ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);
  funcPassManager.addPass(
      createReorderWorkgroups(reorderStrategy, canReorderWorkgroups));

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  funcPassManager.addPass(
      createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType::MMA_SYNC));
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  // funcPassManager.addPass(createTransferReadSubviewFusionPass());
  // funcPassManager.addPass(createFuseForallPass());

  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Distribute shared memory copies.
  funcPassManager.addPass(createVectorExtTransferToVectorTransferPass());
  funcPassManager.addPass(createTransferOpCanonicalizePass());
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Vector -> MMA ops
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(
      createLLVMGPUVectorToGPUPass(GPUTensorCoreType::MMA_SYNC));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  // funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  GPUPipeliningPassOptions pipelieningOptions = {};
  pipelieningOptions.epiloguePeeling = false;
  pipelieningOptions.depth = pipelineDepth;
  pipelieningOptions.scheduleIndex =
      llvm::to_underlying(PipeliningSchedulingStrategy::nvidiaTensorCore);
  funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));
  // Optimize shared memory usage. 有毒
  funcPassManager.addPass(createLLVMGPUPackSharedMemoryAllocPass());
}

// //===---------------------------------------------------------------------===//
// // Transpose
// //===---------------------------------------------------------------------===//

// void addGPUTransposePassPipeline(OpPassManager &funcPassManager,
//                                  const GPUPipelineOptions &options) {
//   tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true);

//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   funcPassManager.addPass(
//       createGPUTensorAlloc(GPUPromoteSharedMemPattern::TransposeOpPattern));
//   funcPassManager.addPass(createGPUTensorTilePass());

//   // Linalg -> vector
//   addGPUVectorizationPasses(funcPassManager);
//   funcPassManager.addPass(createOptimizeVectorTransferPass());
//   funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

//   // tensor to memref
//   addBufferizePasses(funcPassManager);

//   // distribute foreach threads
//   funcPassManager.addPass(createGPUDistributePass());

//   funcPassManager.addPass(createMemrefCopyToLinalgPass());
//   funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   if (options.enableReduceSharedMemoryBankConflicts) {
//     // May or may not need to reduce shared mememory conflicts.
//     GPUReduceBankConflictsPassOptions options = {};
//     options.paddingBits = 32;
//     funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
//   }

//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
// }

// //===---------------------------------------------------------------------===//
// // Vector Distribution
// //===---------------------------------------------------------------------===//

// // Matmul pipeline using vector distribution patterns to map to various
// tensor
// // core operations. The current implementation below is unstable and is
// missing
// // a few crucial pieces for performance (primarily software pipelining).
// The
// // current flow is as follows.
// //
// // 1. Tile + fuse and distribute to workgroups.
// // 2. Problem specific tiling, namely tiling the K dimension of the GEMM.
// // 3. Vectorize
// // 4. Materialize shared memory allocations as vectorized copies.
// // 5. Bufferize
// //
// // * Distribution to warps should happen here, but right now this pipeline
// //   is single subgroup. Pending improvements to vector distribution to
// allow
// //   distribution to warps.
// //
// // 6. Distribute to virtual lanes (i.e. threads in this case).
// //
// // Note that a few pieces here are subject to change in the immediate
// future.
// // First, the shared memory promotion done here is in a sense a stopgap, as
// it
// // won't compose well with what's available for bufferization/pipelining
// today.
// // Second, distribution to more than one warp depends on either layout
// changes,
// // or explicit distribution using `scf.forall`. For now this keeps it
// simple
// // and gives us a starting point for generating code for matmuls in the
// first
// // place.

// // We use vector ops to do the copy for this pipeline because distribution
// is
// // vector based.
// static LogicalResult gpuVectorCopyFn(OpBuilder &builder, Location loc,
//                                      Value from, Value to) {
//   bool needsBarrier = false;
//   MemRefType fromType = llvm::cast<MemRefType>(from.getType());
//   if (hasSharedMemoryAddressSpace(fromType)) {
//     needsBarrier = true;
//   }
//   if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(to.getType()))) {
//     needsBarrier = true;
//   }
//   if (needsBarrier)
//     builder.create<gpu::BarrierOp>(loc);
//   VectorType vectorType =
//       VectorType::get(fromType.getShape(), fromType.getElementType());
//   Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
//   SmallVector<Value> indices(vectorType.getRank(), c0);
//   SmallVector<bool> inBounds(vectorType.getRank(), true);
//   Value read = builder.create<vector::TransferReadOp>(loc, vectorType,
//   from,
//                                                       indices, inBounds);
//   builder.create<vector::TransferWriteOp>(loc, read, to, indices,
//   inBounds); if (needsBarrier) {
//     builder.create<gpu::BarrierOp>(loc);
//   }
//   return success();
// }

// static void addVectorBufferizePasses(OpPassManager &funcPassManager) {
//   BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
//   BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
//   addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn,
//   memcpyFn); funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
// }

// void addGPUVectorDistributePassPipeline(OpPassManager &funcPassManager,
//                                         const GPUPipelineOptions &options,
//                                         bool usePadToModelSharedMemcpy) {
//   tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/false);

//   IREE::GPU::ReorderWorkgroupsStrategy reorderStrategy =
//       getReorderWorkgroupsStrategy(options.reorderStrategy);
//   funcPassManager.addPass(
//       createReorderWorkgroups(reorderStrategy, canReorderWorkgroups));

//   if (usePadToModelSharedMemcpy) {
//     funcPassManager.addPass(createLLVMGPUPromoteMatmulToFitMMAPass());
//   }

//   funcPassManager.addPass(
//       IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());

//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
//   funcPassManager.addPass(createGPUPromoteMatmulOperandsPass());

//   // Tile to reduction loops.
//   {
//     GPUApplyTilingLevelPassOptions options;
//     options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
//     options.allowZeroSlices = true;
//     funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
//     funcPassManager.addPass(affine::createLoopCoalescingPass());
//     funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//     funcPassManager.addPass(createCSEPass());
//   }

//   funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   // Set anchors at tensor level for vector distribution later and hoist
//   out
//   // loop invariant anchors.
//   funcPassManager.addPass(createLLVMGPUConfigureTensorLayoutsPass());
//   funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());

//   // Generalize all named ops so that we can fold away unit extent dims. By
//   this
//   // point, all tiling is finished so the tiling configurations on those
//   ops can
//   // be safely dropped. This additionally allows vectorization of
//   convolution to
//   // `vector.contract` as filter dimensions are expected to be tiled to 1
//   by
//   // this point.
//   funcPassManager.addPass(createLinalgGeneralizeNamedOpsPass());
//   if (!usePadToModelSharedMemcpy) {
//     LinalgFoldUnitExtentDimsPassOptions options;
//     options.useRankReducingSlices = true;
//     funcPassManager.addPass(
//         IREE::VectorExt::createVectorExtFoldUnitExtentDimsPass());
//     funcPassManager.addPass(mlir::createLinalgFoldUnitExtentDimsPass(options));
//     funcPassManager.addPass(createCanonicalizerPass());
//     funcPassManager.addPass(createCSEPass());
//   }

//   funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

//   // Linalg -> Vector
//   addGPUVectorizationPasses(funcPassManager);

//   // Allocate tensors for copies to shared memory.
//   funcPassManager.addPass(createGPUVectorAllocPass());
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
//   funcPassManager.addPass(createGPUCombineValueBarriersPass());

//   // Tensor -> Memref
//   addVectorBufferizePasses(funcPassManager);
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
//   funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());

//   // Preprocessing for vector distribution.
//   funcPassManager.addPass(createLLVMGPUCastTypeToFitMMAPass());

//   // Vector SIMD -> Vector SIMT
//   funcPassManager.addPass(createLLVMGPUVectorDistributePass());
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   if (options.enableReduceSharedMemoryBankConflicts) {
//     GPUReduceBankConflictsPassOptions options = {};
//     options.paddingBits = 64;
//     funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
//   }
//   if (options.prefetchSharedMemory) {
//     funcPassManager.addPass(createLLVMGPUPrefetchSharedMemoryPass());
//   }
//   if (clLLVMGPUEnableSharedMemoryReuse) {
//     funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
//     funcPassManager.addPass(createGPUReuseSharedMemoryAllocsPass());
//   }
//   funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
//   funcPassManager.addPass(createCSEPass());
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
// }

void addGPUWarpReductionPassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/false);
  funcPassManager.addPass(createRematerializeParallelOpsPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createGPUTileReductionPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  {
    GenericVectorizationPassOptions options;
    options.enableVectorMasking = true;
    options.useConfiguredVectorSizes = false;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    options.enableCleanup = false;
    options.generateContract = false;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createSPMDOpPass());
  // vector -> simt gpu + vector
  funcPassManager.addPass(createConvertVectorReductionToGPUPass(
      /*expandSubgroupReduction=*/true));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

// void addGPUPackUnPackPasses(OpPassManager &funcPassManager) {
//   tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true);
//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   funcPassManager.addPass(createGPUTensorTilePass());
//   funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   funcPassManager.addPass(createDecomposePackUnPackOpsPass(
//       DecomposePackUnPackOpsPassOptions{/*tileOuterToOne=*/true,
//                                         /*useOnlyReshapes=*/false}));
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());
//   addGPUVectorizationPasses(funcPassManager);

//   addBufferizePasses(funcPassManager);

//   funcPassManager.addPass(createGPUDistributePass());
// }
// }

// void addGPUSimpleDistributePassPipeline(OpPassManager &funcPassManager) {
//   tileAndBufferize(funcPassManager);

//   // Distribute linalg onto threads within the workgroup.
//   funcPassManager.addPass(
//       createLLVMGPUTileAndDistributePass(/*distributeToWarp=*/false));
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   funcPassManager.addPass(createRemoveSingleIterationLoopPass());
// }

// void addGPUDefaultPassPipeline(OpPassManager &funcPassManager,
//                                const GPUPipelineOptions &options) {
//   ConvertToDestinationPassingStylePassOptions dpsOptions;
//   dpsOptions.useWARForCooperativeMatrixCodegen = true;
//   tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/false,
//                                /*convertToDpsOptions=*/dpsOptions);
//   if (options.enableUkernels) {
//     funcPassManager.addPass(createGPULowerToUKernelsPass());
//   }
//   funcPassManager.addPass(createCanonicalizerPass());
//   funcPassManager.addPass(createCSEPass());

//   addBufferizePasses(funcPassManager);
//   funcPassManager.addPass(createRemoveSingleIterationLoopPass());
// }

// // void addGPUBaseLoweringPassPipeline(OpPassManager &funcPassManager) {
// //   funcPassManager.addPass(createConvertToDestinationPassingStylePass(
// //       /*useWARForCooperativeMatrixCodegen=*/false));
// //   funcPassManager.addPass(createCanonicalizerPass());
// //   funcPassManager.addPass(createCSEPass());

// //   addBufferizePasses(funcPassManager);
// //   funcPassManager.addPass(createCanonicalizerPass());
// //   funcPassManager.addPass(createCSEPass());

// // funcPassManager.addPass(IREE::LinalgExt::createLinalgExtToLoopsPass());
// //   funcPassManager.addPass(createMemrefCopyToLinalgPass());
// //   funcPassManager.addPass(createConvertLinalgToLoopsPass());
// //   funcPassManager.addPass(createRemoveSingleIterationLoopPass());
// //   funcPassManager.addPass(createCanonicalizerPass());
// //   funcPassManager.addPass(createCSEPass());
// }

// Add passes to make the address computation more explicit and optimize
// them.
//
// The idea here is to be less dependent on what the LLVM backend is able
// to do,
// by heavy lifting most of the work while we still have the information
// about
// loops.
//
// Note that this needs to run before SCF -> CF.
static void
addLowerAndOptimizeAddressComputationPasses(FunctionLikeNest &funcPassManager) {
  funcPassManager.addPass(createExtractAddressComputationGPUPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      // Hoist loop invariant variables to give affine decomposition pass the
      // right loop dependencies.
      // .addPass(createIREELoopInvariantCodeMotionPass)
      // Decompose affine ops.
      .addPass(createDecomposeAffineOpsPass)
      // Get rid of the redundant computations.
      .addPass(createCSEPass)
      // Hoist the resulting decompositions.
      // .addPass(createIREELoopInvariantCodeMotionPass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass(createLowerAffinePass)
      // Do another round of LICM now that we've lowered and optimized
      // arithmetic
      .addPass(createCSEPass);
  // .addPass(createIREELoopInvariantCodeMotionPass);
}

static void addLowerToLLVMGPUPasses(OpPassManager &modulePassManager,
                                    bool forROCDL) {
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // modulePassManager.addPass(createLowerUKernelOpsToCallsPass());

  FunctionLikeNest(modulePassManager)
      // Linalg -> SCF
      .addPass(createMemrefCopyToLinalgPass)
      .addPass(createConvertLinalgToLoopsPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // Pad allocations with dynamic dimension after linalg lowering but before
      // lowering SCF and affine ops.
      // .addPass(createPadDynamicAllocPass)
      // Hoist any newly static allocations from PadDynamicAlloc.
      .addPass(createHoistStaticallyBoundAllocationsPass)
      .addPass(createLowerAffinePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Handled tensor constants.
  addConstantBufferizePasses(modulePassManager);

  FunctionLikeNest funcPassManager(modulePassManager);
  funcPassManager.addPass(createFoldTensorExtractOpPass)
      .addPass(createLLVMGPUVectorLoweringPass)
      .addPass(createExpandGPUOpsPass)
      // Expose workitem and workgroup counts to range inference later.
      .addPass(createGPUPropagateDispatchSizeBoundsPass);

  // This pass needs to run before SCF -> CF.
  addLowerAndOptimizeAddressComputationPasses(funcPassManager);

  // Run checks on shared memory usage.
  funcPassManager
      .addPass([&]() {
        auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 64; };
        return createGPUCheckResourceUsagePass(getIndexBitwidth);
      })
      // SCF -> CF
      .addPass(createConvertSCFToCFPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // Convert BF16 operations to occur as F32.
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass(createConvertBf16ToUInt16BuffersPass)
      // Convert math dialect elementry functions to polynomial form.
      .addPass(createPolynomialApproximationPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      .addPass(createEmulateNarrowTypePass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass(createLowerAffinePass);

  // Strip out the debug info for the kernel.
  modulePassManager.addPass(createStripDebugInfoPass());
  // Cast address spaces of all function arguments to generic.
  modulePassManager.addPass(createLLVMGPUCastAddressSpaceFunctionPass());
  // convert to NVVM.
  modulePassManager.addPass(createConvertToNVVMPass());
}

// void addGPUTransformDialectPasses(OpPassManager &funcPassManager,
//                                   StringRef entryPoint) {
//   funcPassManager.addPass(
//       mlir::tts::createTransformDialectInterpreterPass(entryPoint));

//   // Dropping the schedule is needed:
//   //   1. if we want to embed the transform in the module: we should drop the
//   //      schedule once applied.
//   //   2. if transform.do_not_dce_operands ops are introduced.
//   funcPassManager.addPass(createDropSchedulePass());
// }

// //===----------------------------------------------------------------------===//
// // Common Pass Pipelines
// //===----------------------------------------------------------------------===//

// static void buildLLVMGPUCodegenConfigurationPassPipelineImpl(
//     OpPassManager &modulePassManager) {
//   {
//     FunctionLikeNest funcPassManager(modulePassManager);
//     funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
//     addCommonTargetExecutablePreprocessingPasses(funcPassManager);
//     addEncodingToNopPasses(funcPassManager);
//     funcPassManager.addPass(createBlockDynamicDimensionsPass);
//     funcPassManager.addPass(createConfigTrackingCanonicalizerPass);
//     funcPassManager.addPass(createCSEPass);
//   }
//   modulePassManager.addPass(createMaterializeTuningSpecsPass());
//   modulePassManager.addPass(createMaterializeUserConfigsPass());
//   modulePassManager.addPass(createLLVMGPUSelectLoweringStrategyPass());
// }

// void buildLLVMGPUCodegenConfigurationPassPipeline(
//     OpPassManager &variantPassManager) {
//   buildLLVMGPUCodegenConfigurationPassPipelineImpl(
//       variantPassManager.nest<ModuleOp>());
// }

void buildLLVMGPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool useROCM) {

  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    FunctionLikeNest(modulePassManager)
        .addPass(createLLVMGPULowerExecutableTargetPass)
        .addPass(createVerifyWorkgroupDistributionPass);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());

  //   //===--------------------------------------------------------------------===//
  //   // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //   //
  //   // Post-conditions:
  //   //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   //   - The module contains the final llvm.module ready to be serialized.
  //   //===--------------------------------------------------------------------===//
  //   addLowerToLLVMGPUPasses(variantPassManager.nest<ModuleOp>(), useROCM);

  LLVM_DEBUG({
    llvm::dbgs() << "Using LLVMGPU pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

// // NOTE: this runs on the top-level program module containing all
// // hal.executable ops.
// void buildLLVMGPULinkingPassPipeline(OpPassManager &modulePassManager,
//                                      std::optional<std::string> target) {
//   // Link together executables. This may produce some IR duplication.
//   LLVMGPULinkExecutablesPassOptions linkOptions;
//   linkOptions.target = target.value_or("");
//   modulePassManager.addPass(createLLVMGPULinkExecutablesPass(linkOptions));

//   // Cleanup IR duplication.
//   modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
//       mlir::createCanonicalizerPass());

//   // Assign final executable constant and import ordinals.
//   auto &variantPassManager =
//   modulePassManager.nest<IREE::HAL::ExecutableOp>()
//                                  .nest<IREE::HAL::ExecutableVariantOp>();
//   variantPassManager.addPass(createLLVMGPUAssignConstantOrdinalsPass());
// }

//===---------------------------------------------------------------------===//
// Common Pass Registration
//===---------------------------------------------------------------------===//

namespace common {
#define GEN_PASS_REGISTRATION
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"
} // namespace common

void registerCodegenLLVMGPUPasses() {
  // Generated.
  common::registerPasses();

  // static PassPipelineRegistration<> LLVMGPUConfigPipeline(
  //     "iree-codegen-llvmgpu-configuration-pipeline",
  //     "Runs the translation strategy configuration pipeline on Linalg for
  //     GPU " "on all functions in a module",
  //     [](OpPassManager &modulePassManager) {
  //       buildLLVMGPUCodegenConfigurationPassPipelineImpl(modulePassManager);
  //     });

  static PassPipelineRegistration<> LinalgNVVMPipeline(
      "iree-codegen-linalg-to-nvvm-pipeline",
      "Runs the progressive lowering pipeline from Linalg to NVVM",
      [](OpPassManager &passManager) {
        buildLLVMGPUCodegenPassPipeline(passManager, false);
      });

  // static PassPipelineRegistration<> LinalgROCDLPipeline(
  //     "iree-codegen-linalg-to-rocdl-pipeline",
  //     "Runs the progressive lowering pipeline from Linalg to ROCDL",
  //     [](OpPassManager &passManager) {
  //       buildLLVMGPUCodegenPassPipeline(passManager, true);
  //     });

  // static PassPipelineRegistration<> LLVMGPULinkingPipeline(
  //     "iree-codegen-llvmgpu-linking-pipeline",
  //     "Runs the LLVMGPU HAL executable linking pipeline",
  //     [](OpPassManager &modulePassManager) {
  //       buildLLVMGPULinkingPassPipeline(modulePassManager);
  //     });

  registerPass(createTensorTransferWriteFusionPass);
}

} // namespace mlir::tts
