get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_triton_library(TTSNVLinalgToLLVM
  LinalgToLLVM.cpp
  MemrefCopyToLinalg.cpp
  
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
  MLIRControlFlowToLLVM
  MLIRSCFToGPU
  MLIRGPUPipelines
  MLIRNVVMToLLVM
  MLIRVectorToGPU
  ${conversion_libs}
  ${extension_libs}
  )
