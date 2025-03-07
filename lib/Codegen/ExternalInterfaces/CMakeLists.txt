
add_triton_library(IREECodegenExternalModels
    GPUEncodingExternalModels.cpp
    Interfaces.cpp

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRIR
    MLIRLinalgDialect
    MLIRTensorDialect
    IREECodegenDialect
    IREECodegenDialectUtils
    IREEGPUDialect
    IREEEncodingDialect
  PUBLIC
)
