add_triton_library(TritonStructuredIR
  TritonStructuredOps.cpp
  TritonStructuredDialect.cpp
  TritonStructuredAttr.cpp

  DEPENDS
  TritonStructuredTableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  MLIRSupport
  MLIRTilingInterface
  MLIRArithDialect
  MLIRArithUtils
  LLVMSupport
  MLIRAffineDialect
  MLIRSideEffectInterfaces
  )
