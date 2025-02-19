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


third_party/llvm-project/build/bin/mlir-opt   .vscode/ttshared.mlir \
 --convert-linalg-to-affine-loops --empty-tensor-to-alloc-tensor --one-shot-bufferize --lower-affine --convert-linalg-to-loops --expand-strided-metadata --lower-affine \
  --convert-vector-to-scf \
  --convert-scf-to-cf   --convert-vector-to-llvm --convert-index-to-llvm --memref-expand --convert-arith-to-llvm --convert-math-to-llvm --convert-complex-to-llvm\
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --convert-gpu-to-nvvm \
  --convert-nvvm-to-llvm  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm --lower-affine --convert-arith-to-llvm --reconcile-unrealized-casts \
  --verify-diagnostics

===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0145 seconds

  ----Wall Time----  ----Name----
    0.0014 (  9.9%)  Parser
    0.0002 (  1.3%)  ConvertLinalgToAffineLoopsPass
    0.0001 (  0.7%)  EmptyTensorToAllocTensor
    0.0002 (  1.2%)  OneShotBufferize
    0.0001 (  0.5%)  ConvertAffineToStandard
    0.0003 (  1.9%)  ConvertLinalgToLoopsPass
    0.0001 (  0.5%)  ExpandStridedMetadata
    0.0000 (  0.3%)  ConvertAffineToStandard
    0.0001 (  0.7%)  ConvertVectorToSCF
    0.0001 (  0.7%)  SCFToControlFlow
    0.0002 (  1.5%)  ConvertVectorToLLVMPass
    0.0001 (  0.6%)  ConvertIndexToLLVMPass
    0.0000 (  0.3%)  ExpandOps
    0.0002 (  1.4%)  ArithToLLVMConversionPass
    0.0001 (  0.5%)  ConvertMathToLLVMPass
    0.0001 (  0.4%)  ConvertComplexToLLVMPass
    0.0000 (  0.2%)  'func.func' Pipeline
    0.0000 (  0.2%)    GpuMapParallelLoopsPass
    0.0000 (  0.3%)  ConvertParallelLoopToGpu
    0.0001 (  0.5%)  ConvertNVVMToLLVMPass
    0.0008 (  5.7%)  FinalizeMemRefToLLVMConversionPass
    0.0000 (  0.1%)    (A) DataLayoutAnalysis
    0.0004 (  3.0%)  ConvertFuncToLLVMPass
    0.0000 (  0.1%)    (A) DataLayoutAnalysis
    0.0002 (  1.5%)  ConvertControlFlowToLLVMPass
    0.0003 (  2.0%)  ConvertAffineToStandard
    0.0002 (  1.5%)  ArithToLLVMConversionPass
    0.0001 (  1.0%)  ReconcileUnrealizedCasts
    0.0078 ( 53.9%)  Output
    0.0011 (  7.6%)  Rest
    0.0145 (100.0%)  Total