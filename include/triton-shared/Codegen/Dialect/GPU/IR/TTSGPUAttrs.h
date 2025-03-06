#ifndef TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUATTRS_H_
#define TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUATTRS_H_

#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenInterfaces.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUDialect.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUEnums.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUInterfaces.h"
#include "triton-shared/Codegen/Utils/VectorOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tts::GPU {

struct MMASingleSubgroupLayout {
  SmallVector<int64_t, 2> outer;
  SmallVector<int64_t, 2> thread;
  SmallVector<int64_t, 2> tstrides;
  SmallVector<int64_t, 2> element;
};

MMASingleSubgroupLayout getSingleSubgroupLayout(MMAIntrinsic intrinsic,
                                                MMAFragment fragment);

MMASingleSubgroupLayout getSingleSubgroupLayout(VirtualMMAIntrinsic intrinsic,
                                                MMAFragment fragment);

MMASingleSubgroupLayout getSingleSubgroupLayout(MmaInterfaceAttr mmaKind,
                                                MMAFragment fragment);

StringRef getTilingLevelName(GPU::TilingLevel level);

} // namespace mlir::tts::GPU

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUAttrs.h.inc"

#endif // TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUATTRS_H_
