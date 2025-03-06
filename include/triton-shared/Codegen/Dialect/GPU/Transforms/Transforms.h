

//===- Transforms.h - Transformations for the IREE GPU dialect ------------===//
//
// Defines transformations that apply to IREE GPU ops for use in multiple
// places.
//
//===----------------------------------------------------------------------===//
#ifndef TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
#define TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_

#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::linalg {
class LinalgOp;
}

namespace mlir::scf {
class ForallOp;
}

namespace mlir::vector {
struct UnrollVectorOptions;
}

namespace mlir::tts::GPU {

/// Function to fuse the given producer-consumer pair of forall loops into
/// the single consumer loop. This is managed by inserting an
/// `tts_gpu.barrier_region` at the boundary to synchronize the workers at
/// the fusion point.
///
/// Copy semantics of tensors means that having multiple threads (i.e. in an
/// scf.forall) inserting into a tensor has unclear semantics without an op
/// to separate contexts with different levels of parallelism. scf.forall
/// does this through its terminator and `tts_gpu.barrier_region` does this
/// by keeping code writing to shared memory in a distinct region. This allows
/// us to always default to private memory when bufferizing.
///
/// The mapping attributes of both the producer and consumer `scf.forall` ops
/// must be in a relative descending order, for example:
///  [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>]
/// or
///  [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
LogicalResult fuseForallIntoConsumer(RewriterBase &rewriter,
                                     scf::ForallOp producer,
                                     scf::ForallOp consumer,
                                     SmallVector<Operation *> consumerChain);

// Helper to convert a contraction-like linalg op to an tts_gpu.multi_mma.
FailureOr<tts::GPU::MultiMmaOp>
convertContractionToMultiMma(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                             tts::GPU::MmaInterfaceAttr mmaKind);

// Helper to distribute a multi_mma op to lanes.
FailureOr<Operation *> distributeMultiMmaOp(
    RewriterBase &rewriter, tts::GPU::MultiMmaOp mmaOp,
    std::optional<SmallVector<int64_t>> workgroupSize = std::nullopt);

// Helper to map all scf.forall ops on lanes.
void mapLaneForalls(RewriterBase &rewriter, Operation *funcOp,
                    bool insertBarrier);

// Various populate pattern methods.
void populateIREEGPUDropUnitDimsPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerMultiMmaPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerBarrierRegionPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerValueBarrierPatterns(RewritePatternSet &patterns);
void populateIREEGPUVectorUnrollPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options);
// Version of unrolling with a preset configuration.
void populateIREEGPUVectorUnrollPatterns(RewritePatternSet &patterns);
void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns);

} // namespace mlir::tts::GPU

#endif // TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
