

add_triton_library(IREECodegenInterfaces
  Interfaces.cpp
  BufferizationInterfaces.cpp
  PartitionableLoopsInterface.cpp
  UKernelOpInterface.cpp

  DEPENDS
    BufferizationInterfaces
    PartitionableLoopsInterface
    ProcessorOpInterfaces
    UKernelOpInterface
    MLIRFunctionInterfaces
    MLIRLinalgStructuredOpsIncGenLib
    iree::compiler::Codegen::Dialect::GPU::Transforms::BufferizationInterfaces
    IREECodegenExternalModels

  LINK_LIBS PUBLIC
    MLIRAMDGPUDialect
    MLIRAffineDialect
    MLIRAffineTransformOps
    MLIRArithTransforms
    MLIRArithValueBoundsOpInterfaceImpl
    MLIRBufferizationDialect
    MLIRBufferizationTransformOps
    MLIRBufferizationTransforms
    MLIRFuncTransforms
    MLIRGPUTransformOps
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransformOps
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRMemRefTransformOps
    MLIRSCFDialect
    MLIRSCFTransformOps
    MLIRSCFTransforms
    MLIRSubsetOpInterface
    MLIRSupport
    MLIRTensorDialect
    MLIRTensorTransformOps
    MLIRTensorTransforms
    MLIRTransformLoopExtension
    MLIRVectorDialect
    MLIRVectorTransformOps
    MLIRVectorTransforms
    IREECodegenDialectUtils
    TSSIREEGPUDialect
)
