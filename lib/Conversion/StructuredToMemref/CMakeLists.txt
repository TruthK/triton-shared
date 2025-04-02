add_triton_library(StructuredToMemref
  StructuredToMemref.cpp
  StructuredToMemrefPass.cpp
  SimplifyTTSMakeTPtr.cpp
  SimplifyTTSMakeTPtrPass.cpp
  HoistTensorEmptyFromLoopsPass.cpp
  
  DEPENDS
  StructuredToMemrefConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
  TTSUtils
)
