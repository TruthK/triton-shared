#ifndef TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUATTRS
#define TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUATTRS

include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenInterfaces.td"
include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUDialect.td"
include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUEnums.td"
include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUInterfaces.td"
include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def TTSGPU_IteratorTypeEnum
    : EnumAttr<TTSGPU_Dialect, IteratorType, "iterator_type"> {
    let assemblyFormat = "`<` $value `>`";
}

def TTSGPU_IteratorTypeArrayAttr
    : TypedArrayAttrBase<TTSGPU_IteratorTypeEnum,
                         "Iterator type should be an enum.">;

//===----------------------------------------------------------------------===//
// GPU Specific Lowering Config Attributes
//===----------------------------------------------------------------------===//

def TTSGPU_LoweringConfigAttr :
    AttrDef<TTSGPU_Dialect, "LoweringConfig", [
      DeclareAttrInterfaceMethods<TTSCodegen_LoweringConfigAttrInterface, [
        "getWorkgroupTileSizes",
        "getStaticTilingLevelSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
        "hasWorkgroupTilingLevel",
      ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = "drive lowering of an operation for gpu compilation.";
  let description = [{
    GPU specific implementation of a lowering config. This carries just a
    dictionary attribute to store any relevant fields. This is the simplest
    form of a lowering config, offering flexibility at the cost of structure.
  }];

  let assemblyFormat = "`<` $attributes `>`";

  let parameters = (ins
    AttrParameter<"DictionaryAttr",
        "The configured fields, including tiling levels">:$attributes
  );
}

def TTSGPU_DerivedThreadConfig :
    AttrDef<TTSGPU_Dialect, "DerivedThreadConfig", [
      DeclareAttrInterfaceMethods<TTSCodegen_LoweringConfigAttrInterface, [
        "getStaticTilingLevelSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
      ]>
    ]> {
  let mnemonic = "derived_thread_config";
  let summary = [{
    drive lowering of an operation by deriving thread distribution when needed.
  }];
  let description = [{
    Lowering config for a single thread tiling level that is inferred after
    previous (often reduction) levels of tile + fuse. This is intended for
    fused operations where it is much easier to compute the tile sizes to use
    after previous levels of tile + fuse, rather than trying to pre-propagate
    tiling configs.
  }];
  let assemblyFormat = "";
  let parameters = (ins);
}

//===----------------------------------------------------------------------===//
// GPU Workgroup Processor (WGP) Level Feature/Limit Attributes
//===----------------------------------------------------------------------===//

def TTSGPU_ComputeBitwidthsAttr : EnumAttr<
  TTSGPU_Dialect, TTSGPU_ComputeBitwidths, "compute_bitwidths"> {
  let cppNamespace = "::mlir::tts::GPU";
  let assemblyFormat = "$value";
}

def TTSGPU_StorageBitwidthsAttr : EnumAttr<
  TTSGPU_Dialect, TTSGPU_StorageBitwidths, "storage_bitwidths"> {
  let cppNamespace = "::mlir::tts::GPU";
  let assemblyFormat = "$value";
}

def TTSGPU_SubgroupOpsAttr : EnumAttr<
  TTSGPU_Dialect, TTSGPU_SubgroupOps, "subgroup_ops"> {
  let cppNamespace = "::mlir::tts::GPU";
  let assemblyFormat = "$value";
}

def TTSGPU_DotProductOpsAttr : EnumAttr<
  TTSGPU_Dialect, TTSGPU_DotProductOps, "dotproduct_ops"> {
  let cppNamespace = "::mlir::tts::GPU";
  let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// Base MMA vector layout
//===----------------------------------------------------------------------===//

class TTSGPU_MmaVectorLayoutAttr<string attrname, string mmaintrinsic> :
    AttrDef<TTSGPU_Dialect, attrname, [
  DeclareAttrInterfaceMethods<TTSGPU_MmaInterfaceAttr, [
    "getABCElementTypes",
    "getABCVectorTypes",
    "getMNKShape",
    "getSubgroupSize",
    "getMmaScope",
    "buildMmaOperation",
    "populateOperandOffsetsSizesStrides",
  ]>
]> {
  let cppNamespace = "::mlir::tts::GPU";

  string baseDescription = [{
    Attribute describing a particular shape of matrix-multiply and accumulate
    instruction. Abstractly, all attributes of this type represent the following
    unit of arithmetic for matrices A, B, and C.

    ```
      C += A x B
    ```

    Where the shape of matrix `A` is `[m, k]`, `B` is `[k, n]`, and
    `C` is `[m, n]`. This intentionally leaves the layout information abstract
    and uses interface methods to materialize layout information only when
    needed. The shape of the mma intrinsic is stored explicitly here as that
    information is queried frequently.

    The element types for this particular mma intrinsic are |aType|, |bType|,
    and |cType| for matrices `A`, `B`, and `C` respectively.
  }];

  let parameters = (ins
    mmaintrinsic:$intrinsic,
    "int64_t":$mSize,
    "int64_t":$nSize,
    "int64_t":$kSize,
    "::mlir::Type":$aType,
    "::mlir::Type":$bType,
    "::mlir::Type":$cType
  );
}

//===----------------------------------------------------------------------===//
// MMA intrinsic
//===----------------------------------------------------------------------===//

class TTSGPU_MmaEnumAttr<EnumAttrInfo enumInfo, string name = "">
  : EnumAttr<TTSGPU_Dialect, enumInfo, name>;

def TTSGPU_MMAIntrinsicAttr
  : TTSGPU_MmaEnumAttr<TTSGPU_MMAIntrinsic, "mma_intrinsic">;

def TTSGPU_MMAAttr : TTSGPU_MmaVectorLayoutAttr<"MMA", "MMAIntrinsicAttr"> {
  let mnemonic = "mma_layout";
  let cppNamespace = "::mlir::tts::GPU";

  let description = !strconcat(baseDescription, [{
    This mma variant describes configurations for MMA ops. The |intrinsic|
    field specifies which particular MMA intrinsic this refers to, with each
    intrinsic implicating a specific MNK shape and operand types.
  }]);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "MMAIntrinsic":$intrinsic)>
  ];

  let extraClassDeclaration = [{
    int64_t getBlockSize() const;
    SmallVector<VirtualMMAIntrinsic> getVirtualIntrinsics() const;
  }];
}

def TTSGPU_DataTiledMMAAttr :
    AttrDef<TTSGPU_Dialect, "DataTiledMMA", [
  DeclareAttrInterfaceMethods<TTSGPU_MmaInterfaceAttr, [
    "getABCElementTypes",
    "getABCVectorTypes",
    "getMNKShape",
    "getSubgroupSize",
    "getMmaScope",
    "populateOperandOffsetsSizesStrides",
    "buildMmaOperation",
  ]>
]> {
  let mnemonic = "data_tiled_mma_layout";
  let cppNamespace = "::mlir::tts::GPU";

  let description = [{
    This mma variant represents MMA ops with data-tiling details. The
    |intrinsic| field specifies which particular MMA intrinsic is targeted by
    the data-tiling.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    "::mlir::tts::GPU::MMAIntrinsicAttr":$intrinsic,
    DefaultValuedParameter<"int64_t", "1">:$unroll_m,
    DefaultValuedParameter<"int64_t", "1">:$subgroups_m,
    DefaultValuedParameter<"int64_t", "1">:$unroll_n,
    DefaultValuedParameter<"int64_t", "1">:$subgroups_n,
    DefaultValuedParameter<"int64_t", "1">:$unroll_k
  );
}

def TTSGPU_VirtualMMAIntrinsicAttr
  : TTSGPU_MmaEnumAttr<TTSGPU_VirtualMMAIntrinsic, "virtual_mma_intrinsic">;

def TTSGPU_VirtualMMAAttr :
    AttrDef<TTSGPU_Dialect, "VirtualMMA", [
  DeclareAttrInterfaceMethods<TTSGPU_MmaInterfaceAttr, [
    "getABCElementTypes",
    "getABCVectorTypes",
    "getMNKShape",
    "getSubgroupSize",
    "getMmaScope",
    "populateOperandOffsetsSizesStrides",
    "buildMmaOperation",
  ]>
]> {
  let mnemonic = "virtual_mma_layout";
  let cppNamespace = "::mlir::tts::GPU";

  let description = [{
    This mma variant represents "virtual" MMA ops that has modification to
    its native layouts by unrollK and/or interleave reads.
  }];

  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    AttrBuilder<(ins "VirtualMMAIntrinsic":$intrinsic)>
  ];

  let parameters = (ins
    "::mlir::tts::GPU::VirtualMMAIntrinsicAttr":$intrinsic
  );

  let extraClassDeclaration = [{
    int64_t getBlockSize() const;
    int64_t getUnrollK() const;
  }];
}

def TTSGPU_MMAOpsArrayAttr : ArrayOfAttr<
  TTSGPU_Dialect, "MMAOpsArray", "mma_ops", "MMAAttr"> {
  let cppNamespace = "::mlir::tts::GPU";
}

//===----------------------------------------------------------------------===//
// MMA schedule
//===----------------------------------------------------------------------===//

def TTSGPU_MmaScheduleAttr : AttrDef<TTSGPU_Dialect, "MMASchedule"> {
  let mnemonic = "mma_schedule";
  let cppNamespace = "::mlir::tts::GPU";

  let parameters = (ins
    "::mlir::tts::GPU::MmaInterfaceAttr":$intrinsic,
    "int64_t":$subgroup_m_count,
    "int64_t":$subgroup_n_count
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// tts_gpu.gpu_encoding_layout
//===----------------------------------------------------------------------===//

def TTSGPU_GPUEncodingLayoutAttr :
    AttrDef<TTSGPU_Dialect, "GPUEncodingLayout"> {
  let mnemonic = "gpu_encoding_layout";
  let summary = "The encoding layout attribute for GPU backend";
  let description = [{
    This attribute can implement any layout interface methods for data-tiling.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    OptionalParameter<"::mlir::tts::GPU::TargetAttr">:$targetAttr
  );
}

//===----------------------------------------------------------------------===//
// Workgroup processor level description
//===----------------------------------------------------------------------===//

def TTSGPU_TargetWgpAttr : AttrDef<TTSGPU_Dialect, "TargetWgp"> {
  let summary = "Workgroup processor level target description";
  let mnemonic = "target_wgp";
  let cppNamespace = "::mlir::tts::GPU";

  let parameters = (ins
    "ComputeBitwidthsAttr":$compute,
    "StorageBitwidthsAttr":$storage,
    "SubgroupOpsAttr":$subgroup,
    "DotProductOpsAttr":$dot,
    "MMAOpsArrayAttr":$mma,
    "DenseI32ArrayAttr":$subgroup_size_choices,
    "DenseI32ArrayAttr":$max_workgroup_sizes,
    "int32_t":$max_thread_count_per_workgroup,
    "int32_t":$max_workgroup_memory_bytes,
    "DenseI32ArrayAttr":$max_workgroup_counts,
    OptionalParameter<"std::optional<int32_t>">:$max_load_instruction_bits,
    OptionalParameter<"std::optional<int32_t>">:$simds_per_wgp,
    OptionalParameter<"std::optional<int32_t>">:$vgpr_space_bits,
    OptionalParameter<"DictionaryAttr">:$extra
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// GPU Chip Level Feature/Limit Attributes
//===----------------------------------------------------------------------===//

def TTSGPU_TargetChipAttr : AttrDef<TTSGPU_Dialect, "TargetChip"> {
  let summary = "Chip level target description";
  let mnemonic = "target_chip";
  let cppNamespace = "::mlir::tts::GPU";

  let parameters = (ins
    "uint32_t":$wgp_count,
    OptionalParameter<"DictionaryAttr">:$extra
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// GPU Target Attributes
//===----------------------------------------------------------------------===//

def TTSGPU_TargetAttr : AttrDef<TTSGPU_Dialect, "Target"> {
  let summary = "Full GPU target attribute";
  let mnemonic = "target";
  let cppNamespace = "::mlir::tts::GPU";

  let parameters = (ins
    StringRefParameter<"target architecture">:$arch,
    StringRefParameter<"target features">:$features,
    "TargetWgpAttr":$wgp,
    OptionalParameter<"TargetChipAttr">:$chip
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    int getMinSubgroupSize() const {
      return *llvm::min_element(getWgp().getSubgroupSizeChoices().asArrayRef());
    }
    int getMaxSubgroupSize() const {
      return *llvm::max_element(getWgp().getSubgroupSizeChoices().asArrayRef());
    }
    int getPreferredSubgroupSize() const {
      return *llvm::min_element(getWgp().getSubgroupSizeChoices().asArrayRef());
    }

    bool supportsSubgroupShuffle() const {
      return bitEnumContainsAll(getWgp().getSubgroup().getValue(),
                                SubgroupOps::Shuffle);
    }

    bool isAMD() const {
      return getArch().starts_with("gfx") || getArch().starts_with("rdna");
    }
    bool isApple() const { return getArch().starts_with("apple"); }
    bool isARM() const { return getArch().starts_with("valhall"); }
    bool isNVIDIA() const { return getArch().starts_with("sm_"); }
    bool isQualcomm() const { return getArch().starts_with("adreno"); }

    std::optional<int> getCUDAComputeCapability() const;
    bool supportsTF32InputMMAOps() const;
    bool supportsSyncMMAOps() const;
  }];
}

//===----------------------------------------------------------------------===//
// GPU Lane ID
//===----------------------------------------------------------------------===//

def TTSGPU_LaneIdAttr : AttrDef<TTSGPU_Dialect, "LaneId", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>
  ]> {
  let mnemonic = "lane_id";
  let cppNamespace = "::mlir::tts::GPU";
  let parameters = (ins
    "int64_t":$dim
  );
  let assemblyFormat = "`<` $dim `>`";
}

//===---------------------------------------------------------------------===//
// tts_gpu.ukernel_config
//===---------------------------------------------------------------------===//

def TTSGPU_UKernelConfigAttr  :
    AttrDef<TTSGPU_Dialect, "UKernelConfig", []> {
  let mnemonic = "ukernel_config";
  let summary = "An attribute specifying a ukernel that an op can lower to.";
  let assemblyFormat = "`<` struct(params) `>`";
  let parameters = (ins
       "StringAttr":$name,
       "DictionaryAttr":$def_attrs
  );
}

//===----------------------------------------------------------------------===//
// GPU Pipeline Options
//===----------------------------------------------------------------------===//

def TTSGPU_ReorderWorkgroupsStrategyAttr :
    EnumAttr<TTSGPU_Dialect, TTSGPU_ReorderWorkgroupsStrategy, "reorder_workgroups_strategy"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::tts::GPU";
}

def TTSGPU_GPUPipelineOptionsAttr : AttrDef<TTSGPU_Dialect, "GPUPipelineOptions"> {
  let summary = "GPU pipeline options attribute.";
  let mnemonic = "pipeline_options";
  let cppNamespace = "::mlir::tts::GPU";

    let description = [{
    This attributes describes lowering pipeline specific configuration options:
    * prefetch_shared_memory: Boolean option indicating whether or not to run
      the loop prefetching pass in the lowering pipeline.
    * no_reduce_shared_memory_bank_conflicts: Boolean option indicating whether
      or not to skip the bank conflict reduction pass in the lowering pipeline.
    * reorder_workgroups_strategy: Enum attribute indicating which strategy to
      choose for the workgroup reordering pass. Options are `None`, `Swizzle`,
      and `Transpose`.
  }];


  let parameters = (ins
    OptionalParameter<"BoolAttr">:$prefetch_shared_memory,
    OptionalParameter<"BoolAttr">:$no_reduce_shared_memory_bank_conflicts,
    OptionalParameter<"BoolAttr">:$use_igemm_convolution,
    OptionalParameter<"ReorderWorkgroupsStrategyAttr">:$reorder_workgroups_strategy
  );

  let builders = [
    AttrBuilder<(ins
        CArg<"bool", "false">:$prefetch_shared_memory,
        CArg<"bool", "false">:$no_reduce_shared_memory_bank_conflicts,
        CArg<"bool", "false">:$use_igemm_convolution,
        CArg<"std::optional<ReorderWorkgroupsStrategy>", "{}">:$reorder_workgroups_strategy)>
  ];

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // Returns the key name for GPUPipelineOptionsAttr in the translation info
    // config dictionary.
    static StringRef getDictKeyName() {
      return "gpu_pipeline_options";
    }
  }];
}

#endif 
