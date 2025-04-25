add_triton_library(TritonToLinalgExperimental
  TritonToLinalgExperimentalPass.cpp
  TritonStructuredToMemref.cpp
  LinalgGenericFusion.cpp
  ConvertTTSTransferOp.cpp
  LoopUnroll.cpp
  ReinterpretCastHoist.cpp

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
  MLIRLinalgTransforms
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
