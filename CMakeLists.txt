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

if (TRITON_SHARED_BUILD_TTS_NV_BACKEND)
    add_triton_plugin(TritonTTSNV ${CMAKE_CURRENT_SOURCE_DIR}/triton_tts_nv.cc LINK_LIBS TritonSharedAnalysis TritonToLinalg TritonTilingExtIR)
endif()
