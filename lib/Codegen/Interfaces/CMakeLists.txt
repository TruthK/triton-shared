add_triton_library(IREECodegenInterfaces
    Interfaces.cpp
  DEPENDS
    IREECodegenBufferizationInterfaces
    IREECodegenPartitionableLoopsInterface
    IREECodegenUKernelOpInterface
  LINK_LIBS PUBLIC
    MLIRAMDGPUDialect
    MLIRAffineDialect
    MLIRAffineTransformOps
    MLIRArithValueBoundsOpInterfaceImpl
    MLIRBufferizationTransformOps
    MLIRGPUTransformOps
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransformOps
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRMemRefTransformOps
    MLIRSCFDialect
    MLIRSCFTransformOps
    MLIRTensorDialect
    MLIRTensorTransformOps
    MLIRTensorTransforms
    MLIRTransformLoopExtension
    MLIRVectorDialect
    MLIRVectorTransformOps
    MLIRVectorTransforms
    IREECodegenExternalModels
)

add_triton_library(IREECodegenBufferizationInterfaces
    BufferizationInterfaces.cpp

  LINK_LIBS PUBLIC
    MLIRArithTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRFuncTransforms
    MLIRIR
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRSCFTransforms
    MLIRSubsetOpInterface
    MLIRSupport
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRVectorTransforms
    IREEGPUDialectBufferizationInterfaces
    IREECodegenUtils
)

add_triton_library(IREECodegenUKernelOpInterface
    UKernelOpInterface.cpp
  DEPENDS
    UKernelOpInterfaceGen
  LINK_LIBS PUBLIC
    MLIRFunctionInterfaces
    MLIRIR
)


add_triton_library(IREECodegenPartitionableLoopsInterface
    PartitionableLoopsInterface.cpp
  DEPENDS
    PartitionableLoopsInterfaceGen
  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRIR
    MLIRLinalgDialect
    MLIRSupport
    MLIRTensorDialect
    IREEGPUDialect
    )
