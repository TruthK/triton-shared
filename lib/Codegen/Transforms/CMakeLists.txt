add_triton_library(IREECodegenTransforms

    AffineMinDistributedSCFCanonicalization.cpp
    RemoveSingleIterationLoop.cpp
    Transforms.cpp

   LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRAnalysis
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFTransforms
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
    MLIRValueBoundsOpInterface
    MLIRVectorDialect
    IREECodegenDialect
    IREECodegenUtils
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
