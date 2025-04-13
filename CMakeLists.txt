

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) 


add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

if (TRITON_BUILD_PYTHON_MODULE)
    add_triton_plugin(TritonTTSNV ${CMAKE_CURRENT_SOURCE_DIR}/triton_ttsnv.cc 
    LINK_LIBS 
    
    ${dialect_libs}
    TritonToLinalgExperimental
    IREECodegenLLVMGPU
    IREECodegenCommon
    IREECodegenInterfaces
    )
    target_link_libraries(TritonTTSNV PRIVATE Python3::Module pybind11::headers)
endif()
