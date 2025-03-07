add_triton_library(IREEDialectEncoding
  EncodingAttrs.cpp
  EncodingDialect.cpp
  EncodingOps.cpp

  DEPENDS
    EncodingEnumsGen
    EncodingOpsIncGen
    EncodingInterfacesGen
    EncodingTypesGen  

  LINK_LIBS PUBLIC
   LLVMSupport
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRArithUtils
    MLIRControlFlowInterfaces
    MLIRDestinationStyleOpInterface
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTensorDialect
    MLIRTensorUtils
    MLIRTilingInterface
    MLIRViewLikeInterface
  )
