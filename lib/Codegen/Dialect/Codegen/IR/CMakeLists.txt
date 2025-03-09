add_triton_library(IREECodegenDialect
  IREECodegenAttrs.cpp
  IREECodegenDialect.cpp
  IREECodegenInterfaces.cpp
  IREECodegenLibraryManager.cpp
  IREECodegenOps.cpp
  UKernelOps.cpp

  DEPENDS
    IREECodegenDialectGen
    IREECodegenOpsGen
    LoweringConfigGen
    LoweringConfigInterfaceGen
    UKernelOpsGen

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRArithDialect
    MLIRArithUtils
    MLIRBufferizationDialect
    MLIRDestinationStyleOpInterface
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRMemRefDialect
    MLIRParser
    MLIRSCFDialect
    MLIRSupport
    MLIRTransformDialect
    MLIRTransformDialectTransforms
    MLIRViewLikeInterface
    IREECodegenUKernelOpInterface
)
