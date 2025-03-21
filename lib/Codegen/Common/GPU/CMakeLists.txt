add_triton_library(IREECodegenCommonGPUPasses
    GPUPatterns.cpp
    GPUMultiBuffering.cpp
    Passes.cpp
    WorkgroupReordering.cpp

  DEPENDS
    CommonGPUPassIncGen

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAMDGPUDialect
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRAnalysis
    MLIRArithDialect
    MLIRArithUtils
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRDestinationStyleOpInterface
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRGPUDialect
    MLIRGPUTransformOps
    MLIRGPUTransforms
    MLIRGPUUtils
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRLoopLikeInterface
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRNVGPUDialect
    MLIRPass
    MLIRRewrite
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSCFUtils
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRTilingInterface
    MLIRTransformUtils
    MLIRTransforms
    MLIRValueBoundsOpInterface
    MLIRVectorDialect
    MLIRVectorToSCF
    MLIRVectorTransforms
    IREECodegenCommon
    IREECodegenDialect
    IREECodegenDialectUtils
    IREEGPUDialect
    IREEVectorExtDialect
    IREECodegenPartitionableLoopsInterface
    IREECodegenTransforms
    IREECodegenUtils
    IREECodegenVectorOpUtils
)

# iree::compiler::Codegen::Common::VectorLayoutAnalysis
# iree::compiler::Codegen::Dialect::GPU::Transforms::GPUTransforms