add_triton_library(IREEGPUDialectBufferizationInterfaces
    BufferizationInterfaces.cpp
LINK_LIBS PUBLIC
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRGPUDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSupport
    IREEGPUDialect
)

add_triton_library(IREEGPUDialectTransforms
    CombineBarrierRegions.cpp
    ConcretizeMmaShapes.cpp
    DistributeMmaToLanes.cpp
    LowerIREEGPUOps.cpp
    Passes.cpp
    Transforms.cpp
    UnrollToIntrinsics.cpp
    VectorizeIREEGPUOps.cpp
    GPUDistributeForall.cpp

  DEPENDS
  GPUTransformsPassesIncGen

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRDestinationStyleOpInterface
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRGPUDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLoopLikeInterface
    MLIRPass
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRVectorUtils
    IREECodegenDialect
    IREEGPUDialect
    IREEVectorExtDialect
    IREECodegenUtils
)

# iree::compiler::Codegen::Transforms