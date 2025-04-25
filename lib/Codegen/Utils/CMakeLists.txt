add_triton_library(IREECodegenUtils
    GPUUtils.cpp
    LinalgOpInfo.cpp
    MarkerUtils.cpp
    Utils.cpp
    ConversionUtils.cpp
    
  LINK_LIBS PUBLIC
    LLVMSupport
    LLVMTargetParser
    MLIRAffineDialect
    MLIRAnalysis
    MLIRArithDialect
    MLIRArithUtils
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRGPUDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTensorDialect
    MLIRTilingInterface
    MLIRTransformUtils
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRViewLikeInterface
    IREECodegenDialect
    IREEGPUDialect
    IREEGPUKnownTargets
    IREECodegenPartitionableLoopsInterface
    IREECodegenUKernelOpInterface

    TritonStructuredIR
)

add_triton_library(IREECodegenVectorOpUtils
  VectorOpUtils.cpp

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRIR
    MLIRLinalgDialect
    MLIRSupport
    MLIRVectorDialect
)