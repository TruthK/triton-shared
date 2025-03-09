add_triton_library(IREECodegenDialectUtils
  Utils.cpp

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRArithDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRTensorDialect
    IREECodegenDialect
    IREEEncodingDialect
)