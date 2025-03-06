#ifndef TTS_COMPILER_CODEGEN_DIALECT_GPU_TTSGPUENUMS
#define TTS_COMPILER_CODEGEN_DIALECT_GPU_TTSGPUENUMS

include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// GPU Workgroup Processor (WGP) Level Feature/Limit Enums
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Compute

def TTSGPU_CFBW_64 : I32BitEnumAttrCaseBit<"FP64", 0, "fp64">;
def TTSGPU_CFBW_32 : I32BitEnumAttrCaseBit<"FP32", 1, "fp32">;
def TTSGPU_CFBW_16 : I32BitEnumAttrCaseBit<"FP16", 2, "fp16">;
def TTSGPU_CIBW_64 : I32BitEnumAttrCaseBit<"Int64", 3, "int64">;
def TTSGPU_CIBW_32 : I32BitEnumAttrCaseBit<"Int32", 4, "int32">;
def TTSGPU_CIBW_16 : I32BitEnumAttrCaseBit<"Int16", 5, "int16">;
def TTSGPU_CIBW_8  : I32BitEnumAttrCaseBit<"Int8",  6, "int8">;

def TTSGPU_ComputeBitwidths : I32BitEnumAttr<
  "ComputeBitwidths", "Supported bitwidths for compute",
  [TTSGPU_CFBW_64, TTSGPU_CFBW_32, TTSGPU_CFBW_16,
   TTSGPU_CIBW_64, TTSGPU_CIBW_32, TTSGPU_CIBW_16, TTSGPU_CIBW_8]> {
  let cppNamespace = "::mlir::tts::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Storage

def TTSGPU_SBW_64 : I32BitEnumAttrCaseBit<"B64", 0, "b64">;
def TTSGPU_SBW_32 : I32BitEnumAttrCaseBit<"B32", 1, "b32">;
def TTSGPU_SBW_16 : I32BitEnumAttrCaseBit<"B16", 2, "b16">;
def TTSGPU_SBW_8  : I32BitEnumAttrCaseBit<"B8",  3, "b8">;

def TTSGPU_StorageBitwidths : I32BitEnumAttr<
  "StorageBitwidths", "Supported bitwidths for storage",
  [TTSGPU_SBW_64, TTSGPU_SBW_32, TTSGPU_SBW_16, TTSGPU_SBW_8]> {
  let cppNamespace = "::mlir::tts::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Subgroup operations

def TTSGPU_SO_None       : I32BitEnumAttrCaseNone<"None", "none">;
def TTSGPU_SO_Shuffle    : I32BitEnumAttrCaseBit<"Shuffle",    0, "shuffle">;
def TTSGPU_SO_Arithmetic : I32BitEnumAttrCaseBit<"Arithmetic", 1, "arithmetic">;

def TTSGPU_SubgroupOps : I32BitEnumAttr<
  "SubgroupOps", "Supported subgroup ops",
  [TTSGPU_SO_None, TTSGPU_SO_Shuffle, TTSGPU_SO_Arithmetic]> {
  let cppNamespace = "::mlir::tts::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Dot product operations

def TTSGPU_DPO_None      : I32BitEnumAttrCaseNone<"None", "none">;
def TTSGPU_DPO_4xI8ToI32 : I32BitEnumAttrCaseBit<"DP4xI8ToI32", 0, "dp4xi8toi32">;

def TTSGPU_DotProductOps : I32BitEnumAttr<
  "DotProductOps", "Supported dot product ops",
  [TTSGPU_DPO_None, TTSGPU_DPO_4xI8ToI32]> {
  let cppNamespace = "::mlir::tts::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// MMA intrinsic

class TTSGPU_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::tts::GPU";
  let genSpecializedAttr = 0;
}

def NV_WMMA_F32_16x16x16_F16 : I32EnumAttrCase<"NV_WMMA_F32_16x16x16_F16", 0x2020>;
def NV_WMMA_F16_16x16x16_F16 : I32EnumAttrCase<"NV_WMMA_F16_16x16x16_F16", 0x2021>;

def TTSGPU_MMAIntrinsic : TTSGPU_I32EnumAttr<"MMAIntrinsic",
    "Descriptor for different MMA intrinsics", [
      NV_WMMA_F32_16x16x16_F16,
      NV_WMMA_F16_16x16x16_F16
    ]>;

def VMFMA_F32_16x16x32_F16  : I32EnumAttrCase<"VMFMA_F32_16x16x32_F16", 0>;
def VMFMA_F32_32x32x16_F16  : I32EnumAttrCase<"VMFMA_F32_32x32x16_F16", 1>;
def VMFMA_F32_16x16x32_F8E4M3FNUZ  : I32EnumAttrCase<"VMFMA_F32_16x16x32_F8E4M3FNUZ", 2>;
def VMFMA_F32_32x32x16_F8E4M3FNUZ  : I32EnumAttrCase<"VMFMA_F32_32x32x16_F8E4M3FNUZ", 3>;

def TTSGPU_VirtualMMAIntrinsic : TTSGPU_I32EnumAttr<"VirtualMMAIntrinsic",
    "Descriptor for different Virtual MMA intrinsics", [
      VMFMA_F32_16x16x32_F16,
      VMFMA_F32_32x32x16_F16,
      VMFMA_F32_16x16x32_F8E4M3FNUZ,
      VMFMA_F32_32x32x16_F8E4M3FNUZ
    ]>;

def MMA_LHS : I32EnumAttrCase<"Lhs", 0>;
def MMA_RHS : I32EnumAttrCase<"Rhs", 1>;
def MMA_ACC : I32EnumAttrCase<"Acc", 2>;

def TTSGPU_MMAFragment : TTSGPU_I32EnumAttr<"MMAFragment",
    "Descriptor for a particular fragment of an MMA operation", [
      MMA_LHS,
      MMA_RHS,
      MMA_ACC
    ]>;

def MMA_Workgroup : I32EnumAttrCase<"Workgroup", 0>;
def MMA_Subgroup : I32EnumAttrCase<"Subgroup", 1>;

def TTSGPU_MMAScope : TTSGPU_I32EnumAttr<"MMAScope",
    "Descriptor for a particular scope of an MMA operation", [
      MMA_Workgroup,
      MMA_Subgroup
    ]>;

//===----------------------------------------------------------------------===//
// Tiling levels

def Workgroup : I32EnumAttrCase<"Workgroup", 0>;
def Reduction : I32EnumAttrCase<"Reduction", 1>;
def PartialReduction : I32EnumAttrCase<"PartialReduction", 2>;
def Thread : I32EnumAttrCase<"Thread", 3>;
def Subgroup : I32EnumAttrCase<"Subgroup", 4>;
def Lane : I32EnumAttrCase<"Lane", 5>;

def TTSGPU_TilingLevel : TTSGPU_I32EnumAttr<"TilingLevel",
    "Descriptor for tiling levels for GPU lowering configs", [
      Workgroup,
      Reduction,
      PartialReduction,
      Thread,
      Subgroup,
      Lane
    ]>;

//===----------------------------------------------------------------------===//
// Pipeline options

class TTSGPU_I32PipelineEnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::tts::GPU";
  let genSpecializedAttr = 0;
}

def ReorderWorkgroupsNone : I32EnumAttrCase<"None", 0>;
def ReorderWorkgroupsTranspose : I32EnumAttrCase<"Transpose", 1>;

def TTSGPU_ReorderWorkgroupsStrategy : TTSGPU_I32PipelineEnumAttr<"ReorderWorkgroupsStrategy",
    "Strategy for workgroup reordering", [
      ReorderWorkgroupsNone,
      ReorderWorkgroupsTranspose
    ]> {
}

#endif // TTS_COMPILER_CODEGEN_DIALECT_GPU_TTSGPUENUMS
