// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_LLVMGPU_PASSES
#define IREE_CODEGEN_LLVMGPU_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// LLVMGPU Passes (keep alphabetical)
//------------------------------------------------------------------------------

def LLVMGPUCodegenPass : Pass<"llvmgpu-codegen", "ModuleOp"> {
  let summary = "Custom pass wrapping LLVM GPU codegen pipeline";
  let description = [{
    Integrates IREE's LLVM GPU code generation pipeline with configurable ROCm/CUDA backend selection.
  }];
  let options = [
    Option<"computeCapability", "compute-capability",
            "int32_t", /*default*/"89",
            "device compute capability">,
    Option<"ptxVersion", "ptx-version",
            "int32_t", /*default*/"84",
            "PTX version">,
        Option<"numWrap", "num-wrap",
               "int32_t", "4",
               "Number of wrap">
  ];
}


def MaterializeTargetPass : Pass<"iree-materialize-target", "ModuleOp"> {
  let summary = "Materialize the target attribute from the metadata";
   let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"89",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"84",
               "PTX version">,
    ];
  
}

def SPMDOpPass : InterfacePass<"convert-triton-spmdop", "mlir::FunctionOpInterface">{
  let summary = "convert-triton-spmdop";
}


// // TODO: Bring the argument in line with the names used elsewhere.
def ConvertToNVVMPass :
    Pass<"iree-convert-to-nvvm", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and NVVM dialects";
     let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", "89",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", "84",
               "PTX version">,
        Option<"numWrap", "num-wrap",
               "int32_t", "4",
               "Number of wrap">
    ];
}


def ExtractAddressComputationGPUPass: Pass<"extract-address-computation-gpu"> {
  let summary = "Extract address computations from memory accesses";
  let description = [{
     This pass is similar to `extract-address-computation` except it also
     supports memory accesses that are specific to GPUs.
  }];
  let dependentDialects = [
      "memref::MemRefDialect", "nvgpu::NVGPUDialect", "affine::AffineDialect"
  ];
}


def LLVMGPUCastAddressSpaceFunctionPass :
    Pass<"iree-llvmgpu-cast-address-space-function", "ModuleOp"> {
  let summary = "Cast address space to generic in CallOp and FuncOp";
}

def TransferOpCanonicalizePass :
    InterfacePass<"iree-llvmgpu-vector-transfer-read-optimize", "mlir::FunctionOpInterface"> {
  let summary = "Optimize vector.transfer_read operations by removing unnecessary bufferization";
  let description = [{
    This pass optimizes vector.transfer_read operations by removing the pattern:
    %tensor = bufferization.to_tensor %memref
    %memref2 = bufferization.to_memref %tensor
    %result = vector.transfer_read %memref2[...]
    
    And replacing it with:
    %result = vector.transfer_read %memref[...]
  }];
  let dependentDialects = [
      "vector::VectorDialect", "bufferization::BufferizationDialect", "memref::MemRefDialect","mlir::tts::IREE::VectorExt::IREEVectorExtDialect"
  ];
}

// def LLVMGPUCastTypeToFitMMAPass : InterfacePass<"iree-llvmgpu-cast-type-to-fit-mma",
//                                                 "mlir::FunctionOpInterface"> {
//   let summary = "Perform type extension/truncation over vector.contract types "
//                 "to target GPU MMA intrinsics";
// }

// def LLVMGPUConfigureTensorLayoutsPass :
//     InterfacePass<"iree-llvmgpu-configure-tensor-layouts", "mlir::FunctionOpInterface"> {
//   let summary = "Pass to set layouts on tensors for later vector distribution";
// }

def LLVMGPULowerExecutableTargetPass :
    InterfacePass<"iree-llvmgpu-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary = "Perform lowering of executable target using one of the DispatchLoweringPassPipeline";
}


def LLVMGPUPackSharedMemoryAllocPass :
    InterfacePass<"iree-llvmgpu-pack-shared-memory-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass pack shared memory allocation in order to reduce memory usage.";
}

def LLVMGPUPrefetchSharedMemoryPass :
    InterfacePass<"iree-llvmgpu-prefetch-shared-memory", "mlir::FunctionOpInterface"> {
  let summary = "Rotate scf.for loops to prefetch shared memory with distance 1";
}

// def LLVMGPUPromoteMatmulToFitMMAPass :
//     InterfacePass<"iree-llvmgpu-promote-matmul-to-fit-mma", "mlir::FunctionOpInterface"> {
//   let summary = "Pass to promote contraction ops to fit mma shapes";
// }

def LLVMGPUSelectLoweringStrategyPass :
    Pass<"iree-llvmgpu-select-lowering-strategy", "ModuleOp"> {
  let summary = "Select a IREE::HAL::DispatchLoweringPassPipeline for lowering the target variant";
}

def LLVMGPUTensorCoreVectorizationPass :
    InterfacePass<"iree-llvmgpu-tensorcore-vectorization", "mlir::FunctionOpInterface"> {
  let summary = "Pass to convert linalg into Vector and transform it to a form that can be lowered to GPU MMA ops";
}

// def LLVMGPUTensorPadPass :
//     InterfacePass<"iree-llvmgpu-tensor-pad", "mlir::FunctionOpInterface"> {
//   let summary = "Pass to pad out tensors up to static dimensions.";
// }

def LLVMGPUTileAndDistributePass :
    InterfacePass<"iree-llvmgpu-tile-and-distribute", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile and distribute linalg ops within a workgroup.";
}

// def LLVMGPUVectorDistributePass :
//     InterfacePass<"iree-llvmgpu-vector-distribute", "mlir::FunctionOpInterface"> {
//   let summary = "Pass to distribute vectorized functions.";
// }

def LLVMGPUVectorLoweringPass :
    InterfacePass<"iree-llvmgpu-vector-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower Vector ops before conversion to LLVM.";
}

def LLVMGPUVectorToGPUPass :
    InterfacePass<"iree-llvmgpu-vector-to-gpu", "mlir::FunctionOpInterface"> {
  let summary = "Pass to convert vector to gpu.";
}

def FuseForallPass :
    InterfacePass<"iree-llvmgpu-fuse-forall", "mlir::FunctionOpInterface"> {
  let summary = "Fuse scf.forall with iree_vector_ext.transfer_write operations.";
}


def TileTTSTransferWritePass :
    InterfacePass<"iree-llvmgpu-tile-tts-transfer-write", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile tts.transfer_write operations.";
}


def TensorTransferWriteFusionPass :
    InterfacePass<"triton-transfer-write-fusion", "mlir::FunctionOpInterface"> {
  let summary = "Fuse tensor.parallel_insert_slice with tts.transfer_write operations";
  let description = [{
    This pass fuses tensor.parallel_insert_slice with tts.transfer_write operations.
    
    When a tts.transfer_write's value comes from an scf.forall result, and the scf.forall
    has only one shared output, and that output is written by tensor.parallel_insert_slice,
    this pass replaces the tensor.parallel_insert_slice with a new tts.transfer_write.
    
    The new tts.transfer_write keeps most information from the original operation,
    but updates indices by adding tensor.parallel_insert_slice's offsets to the original
    tts.transfer_write's indices. The original tts.transfer_write is then deleted.
  }];
  let dependentDialects = [
      "mlir::scf::SCFDialect", "mlir::tts::TritonStructuredDialect", "mlir::tensor::TensorDialect"
  ];
}

//------------------------------------------------------------------------------
// Test Passes
//------------------------------------------------------------------------------

// def TestLLVMGPUScalarizeMathOpPass :
//     Pass<"iree-test-llvmgpu-legalize-ops", "ModuleOp"> {
//   let summary = "Test pass for several legalization patterns.";
// }

// def TestLLVMGPUQueryMMAPass :
//     Pass<"iree-test-llvmgpu-query-mma", "ModuleOp"> {
//   let summary = "Test pass for querying the supported mma intrinsic instructions.";
// }

 def PtrTransformPass :
     Pass<"llvmgpu-ptr-transform", "ModuleOp"> {
   let summary = "Transform !llvm.ptr to !llvm.ptr<1> for matmul_kernel arguments";
   let description = [{
     This pass transforms function signature for matmul_kernel by:
     1. Finding i64 + !llvm.ptr parameter pairs
     2. Removing the i64 parameter
     3. Transforming !llvm.ptr to !llvm.ptr<1>
     4. Creating a new function with the transformed signature
   }];
 }

def SerializeTargetExecutablesPass :
    Pass<"iree-hal-serialize-target-executables", "ModuleOp"> {
  let summary = "Serializes executables for the specified target backend.";
  let description = [{
    Serializes variants for the target backend from their low-level MLIR
    dialects (such as `llvm`, `spirv`, etc) to their target-specific object
    format (static/shared libraries, SPIR-V, etc).
  }];
   let options = [
    Option<"computeCapability", "compute-capability",
            "int32_t", /*default*/"89",
            "device compute capability">,
    Option<"ptxVersion", "ptx-version",
            "int32_t", /*default*/"84",
            "PTX version">,
  ];
}

#endif // IREE_CODEGEN_LLVMGPU_PASSES
