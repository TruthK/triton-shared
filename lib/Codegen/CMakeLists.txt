add_subdirectory(Dialect)
add_subdirectory(ExternalInterfaces)
add_subdirectory(Interfaces)
add_subdirectory(Utils)
add_subdirectory(LLVMGPU)
add_subdirectory(Transforms)
add_subdirectory(Common)


add_triton_library(IREECodegen
    Passes.cpp

LINK_LIBS PUBLIC
    MLIRPass
    IREECodegenDialect
    IREEGPUDialectTransforms
    IREEVectorExtTransforms
    IREECodegenLLVMGPU
    IREECodegenTransforms
    IREEVectorExtDialect
)

# iree::compiler::Codegen::Common
# iree::compiler::Codegen::Common::GPU::CommonGPUPasses
