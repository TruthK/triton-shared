

add_triton_library(IREEDialectVectorExt
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
    iree::compiler::Codegen::Utils::VectorOpUtils
)