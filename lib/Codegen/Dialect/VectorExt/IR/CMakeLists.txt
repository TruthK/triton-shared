add_triton_library(IREEVectorExtDialect
  VectorExtAttrs.cpp
  VectorExtDialect.cpp
  VectorExtOps.cpp

  DEPENDS
    IREEVectorExtAttrsGen
    IREEVectorExtDialectGen
    IREEVectorExtEnumsGen
    IREEVectorExtInterfacesGen
    IREEVectorExtOpsGen

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAffineDialect
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTensorDialect
    MLIRVectorDialect
    IREECodegenVectorOpUtils
)