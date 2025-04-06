add_triton_library(TritonToLinalgExperimental
  TritonToLinalgExperimentalPass.cpp
  TritonStructuredToVector.cpp
  LinalgGenericFusion.cpp

  DEPENDS
  TritonToLinalgExperimentalConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRVectorUtils
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonSharedAnalysis

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
  TritonToUnstructured
  TritonPtrToMemref
  UnstructuredToMemref
  IREEVectorExtDialect
)
