option(TRITON_SHARED_BUILD_TTS_NV_BACKEND "Build triton-shared KZX NV backend" ON)

set(TRITON_SHARED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_SHARED_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Tablegen'd files
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

if (TRITON_SHARED_BUILD_TTS_NV_BACKEND)
    add_triton_plugin(TritonTTSNV ${CMAKE_CURRENT_SOURCE_DIR}/triton_tts_nv.cc 
    LINK_LIBS 
    TritonSharedAnalysis 
    TritonToLinalg 
    TritonTilingExtIR
    TTSUtils
    TTSNVLinalgToLLVM
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ${translation_libs}
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms)
endif()
