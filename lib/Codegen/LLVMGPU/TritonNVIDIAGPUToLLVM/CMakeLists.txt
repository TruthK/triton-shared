add_triton_library(TTSNVIDIAGPUToLLVM
    SPMDOpToLLVM.cpp
    PTXAsmFormat.cpp
    Utility.cpp
    TargetInfo.cpp
    FuncOpToLLVM.cpp
    TypeConverter.cpp


    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRNVGPUDialect
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRNVVMDialect
    MLIRAnalysis
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRLLVMCommonConversion
    MLIRNVVMToLLVM
    MLIRMemRefToLLVM
    MLIRMathToLLVM
    TritonIR
    TritonTransforms
)
