third_party/llvm-project/build/bin/mlir-opt .vscode/ttshared.mlir  --convert-linalg-to-affine-loops --empty-tensor-to-alloc-tensor --one-shot-bufferize --lower-affine --convert-linalg-to-loops --expand-strided-metadata --convert-scf-to-cf --convert-arith-to-llvm --convert-math-to-llvm --convert-complex-to-llvm --convert-vector-to-llvm --convert-index-to-llvm --memref-expand --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --lower-affine --convert-arith-to-llvm --reconcile-unrealized-casts --mlir-print-debuginfo --mlir-timing


----Wall Time----  ----Name----
    0.0030 ( 12.4%)  Parser
    0.0002 (  1.0%)  ConvertLinalgToAffineLoopsPass
    0.0002 (  0.6%)  EmptyTensorToAllocTensor
    0.0002 (  1.0%)  OneShotBufferize
    0.0002 (  0.7%)  ConvertAffineToStandard
    0.0004 (  1.4%)  ConvertLinalgToLoopsPass
    0.0002 (  0.7%)  ExpandStridedMetadata
    0.0001 (  0.4%)  SCFToControlFlow
    0.0002 (  1.0%)  ArithToLLVMConversionPass
    0.0001 (  0.3%)  ConvertMathToLLVMPass
    0.0001 (  0.2%)  ConvertComplexToLLVMPass
    0.0005 (  2.0%)  ConvertVectorToLLVMPass
    0.0001 (  0.4%)  ConvertIndexToLLVMPass
    0.0001 (  0.3%)  ExpandOps
    0.0011 (  4.4%)  FinalizeMemRefToLLVMConversionPass
    0.0000 (  0.1%)    (A) DataLayoutAnalysis
    0.0005 (  2.1%)  ConvertFuncToLLVMPass
    0.0000 (  0.1%)    (A) DataLayoutAnalysis
    0.0004 (  1.5%)  ConvertControlFlowToLLVMPass
    0.0003 (  1.4%)  ConvertAffineToStandard
    0.0004 (  1.6%)  ArithToLLVMConversionPass
    0.0003 (  1.3%)  ReconcileUnrealizedCasts
    0.0101 ( 41.5%)  Output
    0.0058 ( 23.7%)  Rest
    0.0243 (100.0%)  Total
-----------------------------------------------------------------------------------------------


GPUToROCDL
GPUToSPIRV
GPUToVulkan
NVGPUToNVVM
SCFToGPU
VectorToGPU
-------


third_party/llvm-project/build/bin/mlir-opt   .vscode/core_dump_llir.mlir --convert-linalg-to-affine-loops --empty-tensor-to-alloc-tensor --one-shot-bufferize   --convert-linalg-to-loops --lower-affine --expand-strided-metadata   --convert-vector-to-gpu  --gpu-map-parallel-loops  --convert-parallel-loops-to-gpu  --gpu-lower-to-nvvm-pipeline=cubin-chip=sm_89    --convert-scf-to-cf     --convert-vector-to-llvm --convert-index-to-llvm --convert-arith-to-llvm --convert-math-to-llvm --convert-complex-to-llvm   --convert-func-to-llvm  --convert-cf-to-llvm  --convert-arith-to-llvm --convert-nvvm-to-llvm --reconcile-unrealized-casts  --verify-diagnostics

===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.2291 seconds

  ----Wall Time----  ----Name----
    0.0547 ( 23.9%)  Parser
    0.0037 (  1.6%)  ConvertLinalgToAffineLoopsPass
    0.0023 (  1.0%)  EmptyTensorToAllocTensor
    0.0072 (  3.1%)  OneShotBufferize
    0.0062 (  2.7%)  ConvertLinalgToLoopsPass
    0.0017 (  0.8%)  ConvertAffineToStandard
    0.0016 (  0.7%)  ExpandStridedMetadata
    0.0022 (  1.0%)  ConvertVectorToGPU
    0.0007 (  0.3%)  'func.func' Pipeline
    0.0007 (  0.3%)    GpuMapParallelLoopsPass
    0.0010 (  0.4%)  ConvertParallelLoopToGpu
    0.0021 (  0.9%)  ConvertNVGPUToNVVMPass
    0.0008 (  0.4%)  GpuKernelOutlining
    0.0024 (  1.1%)  ConvertVectorToSCF
    0.0028 (  1.2%)  SCFToControlFlow
    0.0017 (  0.7%)  ConvertNVVMToLLVMPass
    0.0110 (  4.8%)  ConvertFuncToLLVMPass
    0.0001 (  0.1%)    (A) DataLayoutAnalysis
    0.0033 (  1.4%)  ExpandStridedMetadata
    0.0015 (  0.7%)  GpuNVVMAttachTarget
    0.0016 (  0.7%)  ConvertAffineToStandard
    0.0016 (  0.7%)  ArithToLLVMConversionPass
    0.0019 (  0.8%)  ConvertIndexToLLVMPass
    0.0030 (  1.3%)  Canonicalizer
    0.0008 (  0.4%)  CSE
    0.0000 (  0.0%)    (A) DominanceInfo
    0.0170 (  7.4%)  GpuToLLVMConversionPass
    0.0028 (  1.2%)  GpuModuleToBinaryPass
    0.0035 (  1.5%)  ConvertMathToLLVMPass
    0.0096 (  4.2%)  Canonicalizer
    0.0033 (  1.4%)  CSE
    0.0000 (  0.0%)    (A) DominanceInfo
    0.0016 (  0.7%)  ReconcileUnrealizedCasts
    0.0058 (  2.5%)  ConvertVectorToSCF
    0.0018 (  0.8%)  SCFToControlFlow
    0.0020 (  0.9%)  ConvertNVVMToLLVMPass
    0.0053 (  2.3%)  ConvertVectorToLLVMPass
    0.0021 (  0.9%)  ConvertIndexToLLVMPass
    0.0023 (  1.0%)  ExpandOps
    0.0023 (  1.0%)  ArithToLLVMConversionPass
    0.0021 (  0.9%)  ConvertMathToLLVMPass
    0.0018 (  0.8%)  ConvertComplexToLLVMPass
    0.0021 (  0.9%)  FinalizeMemRefToLLVMConversionPass
    0.0001 (  0.0%)    (A) DataLayoutAnalysis
    0.0025 (  1.1%)  ConvertFuncToLLVMPass
    0.0001 (  0.0%)    (A) DataLayoutAnalysis
    0.0019 (  0.8%)  ConvertControlFlowToLLVMPass
    0.0020 (  0.9%)  ArithToLLVMConversionPass
    0.0017 (  0.7%)  ReconcileUnrealizedCasts
    0.0136 (  5.9%)  Output
    0.0256 ( 11.2%)  Rest
    0.2291 (100.0%)  Total





third_party/llvm-project/build/bin/mlir-opt   .vscode/core_dump_llir.mlir --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts   --gpu-kernel-outlining --gpu-to-llvm --gpu-to-nvvm --canonicalize --cse