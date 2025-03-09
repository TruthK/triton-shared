add_triton_library(IREEVectorExtTransforms
    Passes.cpp
    VectorExtFoldUnitExtentDims.cpp
    VectorizeIREEVectorExtOps.cpp

    DEPENDS
    VectorExtPassesIncGen

    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRArithDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRVectorUtils
    IREEVectorExtDialect
)
