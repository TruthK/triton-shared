add_triton_library(IREECodegenCommon

    "CleanupBufferAllocViewPass.cpp"
    "ConfigTrackingCanonicalizer.cpp"
    "IREEComprehensiveBufferizePass.cpp"
    "Passes.cpp"
    "ReplaceSlowMinMaxOps.cpp"
    "TileDispatchUsingForall.cpp"
    "TileDispatchUsingInterface.cpp"
DEPENDS
    CommonCodegenPassIncGen
LINK_LIBS PUBLIC
    IREECodegenTransforms
    LLVMSupport
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRAnalysis
    MLIRArithDialect
    MLIRArithTransforms
    MLIRArithUtils
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRBytecodeWriter
    MLIRDestinationStyleOpInterface
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRFunctionInterfaces
    MLIRGPUDialect
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRLoopLikeInterface
    MLIRMathDialect
    MLIRMathTransforms
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRMemRefUtils
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRSCFUtils
    MLIRSideEffectInterfaces
    MLIRSubsetOpInterface
    MLIRSupport
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRTensorUtils
    MLIRTilingInterface
    MLIRTransformDialect
    MLIRTransformUtils
    MLIRTransforms
    MLIRValueBoundsOpInterface
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRViewLikeInterface
    IREECodegenDialect
    IREECodegenDialectUtils
    IREEGPUDialect
    IREEVectorExtDialect
    IREECodegenBufferizationInterfaces
    IREECodegenPartitionableLoopsInterface
    IREECodegenUKernelOpInterface
    IREEEncodingDialect
    IREECodegenUtils
)
#     iree::compiler::Utils

# iree_cc_library(
#   NAME
#     PassHeaders
#   HDRS
#     "Passes.h"
#     "Passes.h.inc"
#   DEPS
#     ::PassesIncGen
#     MLIRLinalgTransforms
#     MLIRMemRefDialect
#     MLIRPass
#     MLIRTransformUtils
#     MLIRTransforms
#     iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
#     iree::compiler::Dialect::HAL::IR
#     iree::compiler::Utils
#   PUBLIC
# )

# iree_tablegen_library(
#   NAME
#     FoldTensorExtractOpIncGen
#   TD_FILE
#     "FoldTensorExtractOp.td"
#   OUTS
#     --gen-rewriters FoldTensorExtractOp.cpp.inc
# )

# iree_cc_library(
#   NAME
#     VectorLayoutAnalysis
#   HDRS
#     "VectorLayoutAnalysis.h"
#   SRCS
#     "VectorLayoutAnalysis.cpp"
#   DEPS
#     LLVMSupport
#     MLIRAnalysis
#     MLIRIR
#     MLIRSCFDialect
#     MLIRVectorDialect
#     iree::compiler::Codegen::Dialect::VectorExt::IR::IREEVectorExtDialect
#   PUBLIC
# )
# iree_cc_library(
#   NAME
#     TransformDialectInterpreterPass
#   SRCS
#     "CommonDialectRegistration.cpp"
#     "MaterializeUserConfigs.cpp"
#     "TransformDialectInterpreterPass.cpp"
#   DEPS
#     ::Common
#     ::PassHeaders
#     ::PassesIncGen
#     IREEDialectsTransforms
#     IREELinalgTransformDialect
#     LLVMSupport
#     MLIRAffineDialect
#     MLIRAffineTransformOps
#     MLIRAffineUtils
#     MLIRAnalysis
#     MLIRArithDialect
#     MLIRArithTransforms
#     MLIRArithUtils
#     MLIRArmSMEDialect
#     MLIRAsyncDialect
#     MLIRBufferizationDialect
#     MLIRBufferizationTransformOps
#     MLIRBufferizationTransforms
#     MLIRFuncDialect
#     MLIRFunctionInterfaces
#     MLIRGPUDialect
#     MLIRGPUTransformOps
#     MLIRIR
#     MLIRLLVMDialect
#     MLIRLinalgDialect
#     MLIRLinalgTransformOps
#     MLIRLinalgTransforms
#     MLIRMemRefTransformOps
#     MLIRPDLDialect
#     MLIRPDLInterpDialect
#     MLIRParser
#     MLIRPass
#     MLIRRewrite
#     MLIRSCFDialect
#     MLIRSCFTransformOps
#     MLIRSCFTransforms
#     MLIRSCFUtils
#     MLIRSupport
#     MLIRTensorDialect
#     MLIRTensorTransformOps
#     MLIRTensorTransforms
#     MLIRTransformDialect
#     MLIRTransformDialectTransforms
#     MLIRTransformLoopExtension
#     MLIRTransformUtils
#     MLIRTransforms
#     MLIRVectorDialect
#     MLIRVectorTransformOps
#     MLIRVectorTransforms
#     iree::compiler::Codegen::Common::TransformExtensions::CommonExtensions
#     iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
#     iree::compiler::Codegen::Dialect::GPU::IR::IREEGPUDialect
#     iree::compiler::Codegen::Dialect::GPU::TransformExtensions::IREEGPUExtensions
#     iree::compiler::Codegen::Dialect::VectorExt::IR::IREEVectorExtDialect
#     iree::compiler::Codegen::LLVMCPU::TransformExtensions::LLVMCPUExtensions
#     iree::compiler::Codegen::LLVMGPU::TransformExtensions::LLVMGPUExtensions
#     iree::compiler::Dialect::Encoding::IR
#     iree::compiler::Dialect::Flow::IR
#     iree::compiler::Dialect::Flow::TransformExtensions::FlowExtensions
#     iree::compiler::Dialect::LinalgExt::IR
#     iree::compiler::Dialect::LinalgExt::TransformExtensions::LinalgExtExtensions
#     iree::compiler::Utils
#   PUBLIC
# )

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
