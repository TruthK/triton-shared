# Exclude tests from libMLIR.so
add_triton_library(TTSNVLinalgToLLVM
  LinalgToLLVM.cpp

  DEPENDS
  TTSUtils
  LinalgToLLVMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRTransforms
  MLIRVectorToLLVMPass
  MLIRVectorToSCF
  MLIRComplexToLLVM
  MLIRComplexToStandard
  MLIRSupport
  )
