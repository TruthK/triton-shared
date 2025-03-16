add_triton_library(TTSUtils
PassUtils.cpp
EmbeddedDataDirectory.cpp

LINK_LIBS PUBLIC
  LLVMSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRTransforms
  MLIRAnalysis
)
