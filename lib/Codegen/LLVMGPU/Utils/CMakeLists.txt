add_triton_library(IREECodegenLLVMGPUUtils
    
    LLVMGPUSelectUKernels.cpp
    LLVMGPUUtils.cpp
    PrefetchSharedMemoryCopy.cpp

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAMDGPUDialect
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRGPUDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRNVGPUDialect
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRVectorDialect
    MLIRAnalysis
    MLIRTransforms
    TTSUtils
    IREECodegenTransformss
    IREEGPUDialect
    IREEVectorExtDialect
    IREECodegenUtils
    IREECodegenVectorOpUtils
)
# iree::compiler::Codegen::Common
# iree::compiler::Codegen::Common::GPU::CommonGPUPasses
# iree::compiler::Codegen::Common::VectorLayoutAnalysis
### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
