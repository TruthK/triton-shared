add_triton_library(IREEGPUHeuristics
   GPUHeuristics.cpp
  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRIR
    MLIRSupport
)

add_triton_library(IREEGPUConfigUtils
    ConfigUtils.cpp
  DEPENDS
    IREEGPUHeuristics
  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRSupport
    IREECodegenDialect
    IREEGPUDialect
    IREECodegenUtils
)

add_triton_library(IREEGPUKnownTargets
    KnownTargets.cpp
  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRIR
    MLIRSupport
    IREEGPUDialect
)
