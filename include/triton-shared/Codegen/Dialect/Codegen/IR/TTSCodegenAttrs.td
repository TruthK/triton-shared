#ifndef TTS_CODEGEN_DIALECT_IREECODEGENATTRS
#define TTS_CODEGEN_DIALECT_IREECODEGENATTRS

include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenDialect.td"
include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenInterfaces.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/EnumAttr.td"

//===---------------------------------------------------------------------===//
// Pass pipelines
//===---------------------------------------------------------------------===//

// List of pre-existing pipelines for translating executables.

def LLVMGPU_Default
    : I32EnumAttrCase<"LLVMGPUDefault", 100>;
def LLVMGPU_BaseLowering
    : I32EnumAttrCase<"LLVMGPUBaseLowering", 101>;
def LLVMGPU_SimpleDistribute
    : I32EnumAttrCase<"LLVMGPUDistribute", 102>;
def LLVMGPU_Vectorize
    : I32EnumAttrCase<"LLVMGPUVectorize", 103>;
def LLVMGPU_MatmulTensorCore
    : I32EnumAttrCase<"LLVMGPUMatmulTensorCore", 104>;
def LLVMGPU_TransposeSharedMem
    : I32EnumAttrCase<"LLVMGPUTransposeSharedMem", 105>;
def LLVMGPU_WarpReduction
    : I32EnumAttrCase<"LLVMGPUWarpReduction", 106>;
def LLVMGPU_PackUnPack
    : I32EnumAttrCase<"LLVMGPUPackUnPack", 107>;
def LLVMGPU_MatmulTensorCoreMmaSync
    : I32EnumAttrCase<"LLVMGPUMatmulTensorCoreMmaSync", 108>;
def LLVMGPU_VectorDistribute
    : I32EnumAttrCase<"LLVMGPUVectorDistribute", 109>;
def LLVMGPU_PadAndVectorDistribute
    : I32EnumAttrCase<"LLVMGPUPadAndVectorDistribute", 110>;
def LLVMGPU_WinogradVectorize
    : I32EnumAttrCase<"LLVMGPUWinogradVectorize", 111>;
def LLVMGPU_TileAndFuse
    : I32EnumAttrCase<"LLVMGPUTileAndFuse", 112>;

def VMVX_Default : I32EnumAttrCase<"VMVXDefault", 300>;

def Linalg_TransformDialectCodegen
    : I32EnumAttrCase<"TransformDialectCodegen", 1000>;
def Custom
    : I32EnumAttrCase<"Custom", 1001>;

def None : I32EnumAttrCase<"None", 0xffff>;

// EnumAttrCase for all known lowerings for ops within dispatch region
// to scalar/native-vector code.
def DispatchLoweringPassPipelineEnum : I32EnumAttr<
  "DispatchLoweringPassPipeline",
  "identifier for pass pipeline use to lower dispatch region", [
    // LLVMGPU CodeGen pipelines
    LLVMGPU_Default, LLVMGPU_BaseLowering, LLVMGPU_SimpleDistribute,
    LLVMGPU_Vectorize, LLVMGPU_MatmulTensorCore,
    LLVMGPU_TransposeSharedMem, LLVMGPU_WarpReduction, LLVMGPU_PackUnPack,
    LLVMGPU_MatmulTensorCoreMmaSync, LLVMGPU_VectorDistribute,
    LLVMGPU_PadAndVectorDistribute, LLVMGPU_WinogradVectorize,
    LLVMGPU_TileAndFuse,

    VMVX_Default,

    // Transform dialect based codegen
    Linalg_TransformDialectCodegen,

    // For out of tree pass-pipelines
    Custom,

    // None to specify no in-built pipelines to use.
    None
  ]> {
  let cppNamespace = "::mlir::tts::GPU";
  // Don't generate a C++ class! We want to use the AttrDef
  let genSpecializedAttr = 0;
}

// Define the AttrDef
def DispatchLoweringPassPipelineAttr :
    EnumAttr<IREECodegen_Dialect, DispatchLoweringPassPipelineEnum, "">;

//===---------------------------------------------------------------------===//
// Triton Codegen workgroup mapping attributes
//===---------------------------------------------------------------------===//

def IdX : I64EnumAttrCase<"IdX", 0, "x">;
def IdY : I64EnumAttrCase<"IdY", 1, "y">;
def IdZ : I64EnumAttrCase<"IdZ", 2, "z">;

def WorkgroupIdEnum :
    I64EnumAttr<"WorkgroupId", "Attribute that map to workgroup.ids", [
      IdX, IdY, IdZ]> {
  let cppNamespace = "::mlir::tts::GPU";
}

def WorkgroupMappingAttr :
    AttrDef<IREECodegen_Dialect, "WorkgroupMapping", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>]> {
  let mnemonic = "workgroup_mapping";

  let parameters = (ins
    EnumParameter<WorkgroupIdEnum>:$id,
    DefaultValuedParameter<"int64_t", "0">:$delinearizedDim
  );
  let assemblyFormat = "`<` $id (`` `:` `` $delinearizedDim^)? `>`";

  let description = [{
    Attribute that eventually will be used to map distributed loop iterations
    to workgroup IDs.

    The `x`,`y` and `z` values for `id` map to workgroup.id[0],
    workgroup.id[1] and workgroup.id[2] respectively.

    In addition it is possible to specify if the `z` dimension is to be
    delinearized on mapping. For example if the list of mapping attributes is
    `[workgroup_mapping<z:1>, workgroup_mapping<z:0>]`, then the `z` dimension
    is delinearized to map to `workgroup_mapping<z:1>` and
    `workgroup_mapping<z:0>`. In other words if the number of logical parallel
    workers along the `z:0` dimension is `W`, then
    ```
    workgroup_mapping<z:0> = workgroup.id[1] mod W,
    workgroup_mapping<z:1> = workgroup.id[1] div W
    ```

    Note: It is expected that this attribute is always used in a list of mapping
    attributes (with a single element being a list of size 1). It is illegal for
    a list to have `workgroup_mapping<z:a>` without `workgroup_mapping<z:b>`
    if `a > b`. In the same way it is illegal to for the list to
    - have `workgroup_mapping<y>` but not `workgroup_mapping<x>`
    - have `workgroup_mapping<z:*>`  but not have `workgroup_mapping<x>`
      and `workgroup_mapping<y>`
  }];
  let builders = [
    AttrBuilder<(ins "WorkgroupId":$id)>
  ];
  let extraClassDeclaration = [{
    // Checks that a list of attributes is well-defined.
    static LogicalResult verifyAttrList(::mlir::MLIRContext *context,
      Location loc, ArrayRef<Attribute> attrs);

    // Convert from mapping ID to attribute.
    static ::mlir::tts::GPU::WorkgroupMappingAttr
    getAttributeFromMappingId(MLIRContext *context, int64_t mappingId);

    // Less than operator for easy comparison.
    bool operator<(
        const ::mlir::tts::GPU::WorkgroupMappingAttr &rhs)
        const;
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// tts_codegen.translation_info
//===---------------------------------------------------------------------===//

def IREECodegen_TranslationInfoAttr :
    AttrDef<IREECodegen_Dialect, "TranslationInfo", []> {
  let mnemonic = "translation_info";
  let summary = [{Drive dispatch entry point lowering}];
  let description = [{
    Specifies the information that is used to drive the translation of
    an entry point function using Linalg based structured-op
    lowering. During executable translation this is attached to the
    export operation.

    If this operation is already set on the root operation (as part of
    compilation info that drives the compilation of a
    dispatch region (like matmul/conv), this attribute gets propagated 
    to the entry point function.

    The fields are:
    - `passPipeline` : The pass pipeline to use.
  }];

  let assemblyFormat = [{
    `<` `pipeline` `=` `` $passPipeline
    (`codegen_spec` `=` $codegenSpec^)?
    (`workgroup_size` `=` `[` $workgroupSize^ `]`)?
    (`subgroup_size` `=` $subgroupSize^)?
    (`,` $configuration^)? `>`
  }];

  let parameters = (ins
    AttrParameter<"tts::GPU::DispatchLoweringPassPipelineAttr",
        "Name of the pipeline to be invoked on the translation unit.">:$passPipeline,
    OptionalParameter<"SymbolRefAttr",
        "The symbol pointing to the transform dialect codegen spec to be used">:$codegenSpec,
    OptionalArrayRefParameter<"int64_t", "The workgroup size to use">:$workgroupSize,
    OptionalParameter<"int64_t", "The subgroup size to use">:$subgroupSize,
    OptionalParameter<"DictionaryAttr",
        "Pipeline specific configuration">:$configuration
  );
  let builders = [
    AttrBuilder<(ins "DispatchLoweringPassPipeline":$passPipeline,
        CArg<"SymbolRefAttr", "{}">:$codegenSpec,
        CArg<"ArrayRef<int64_t>", "{}">:$workgroupSize,
        CArg<"std::optional<int64_t>", "std::nullopt">:$subgroupSize,
        CArg<"DictionaryAttr", "{}">:$configuration)>,
    AttrBuilder<(ins "DispatchLoweringPassPipeline":$passPipeline,
        "ArrayRef<int64_t>":$workgroupSize,
        CArg<"std::optional<int64_t>", "std::nullopt">:$subgroupSize,
        CArg<"DictionaryAttr", "{}">:$configuration)>
  ];
  let extraClassDeclaration = [{
    // Returns the lowering pass pipeline set.
    DispatchLoweringPassPipeline getDispatchLoweringPassPipeline();
  }];
  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// tts_codegen.lowering_config
//===---------------------------------------------------------------------===//

def IREECodegen_LoweringConfigTilingLevelAttr :
  AttrDef<IREECodegen_Dialect, "LoweringConfigTilingLevel", []>
{
  let mnemonic = "lowering_config_level";
  let parameters = (ins
    ArrayRefParameter<"int64_t",
        "The tile sizes to use for this level of tiling">:$sizes,
    OptionalArrayRefParameter<"int64_t",
        "The tile interchange to use for this level of tiling">:$interchange,
    OptionalArrayRefParameter<"bool",
        "The scalable tile flags for this level of tiling">:$scalableFlags);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def IREECodegen_LoweringConfigTilingLevelsAttr :
  ArrayOfAttr<IREECodegen_Dialect, "LoweringConfigTilingLevels",
    "lowering_config_levels", "LoweringConfigTilingLevelAttr", []>
{
}

def IREECodegen_LoweringConfigAttr :
    AttrDef<IREECodegen_Dialect, "LoweringConfig", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getWorkgroupTileSizes",
        "getWorkgroupInterchange",
        "getStaticTilingLevelSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
        "hasWorkgroupTilingLevel",
      ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = [{Drive lowering of an operation within dispatch region}];
  let description = [{
    Default implementation of a lowering configuration attribute. It includes
    only tiling and optionally vectorization information. The interpretation of
    the tiles sizes are backend dependent.

    TODO: Currently there is no verification that the configuration
    specifies everything needed for a pass-pipeline. The values to set
    for these parameters is dependent on the pass-pipeline
    implementation. In future, each pass pipeline could verify that
    the lowering configuration has all the necessary attributes for
    the pipeline.
  }];

  let assemblyFormat = [{
    `<` `tile_sizes` `=` $tilingLevels
      (`,` `native_vector_size` `=` `[` $nativeVectorSize^ `]`)? `>`
  }];

  let parameters = (ins
    AttrParameter<"LoweringConfigTilingLevelsAttr",
        "The lowering config at different levels">:$tilingLevels,
    OptionalArrayRefParameter<"int64_t",
        "The native vector size to use for the given operation">:$nativeVectorSize
  );
  let builders = [
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        CArg<"TileSizesListTypeRef", "{}">:$tileInterchange,
        CArg<"ArrayRef<int64_t>", "{}">:$nativeVectorSize)>,
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        "ScalableTileFlagsListTypeRef":$scalableTileFlags,
        CArg<"TileSizesListTypeRef", "{}">:$tileInterchange,
        CArg<"ArrayRef<int64_t>", "{}">:$nativeVectorSize)>
  ];
  let extraClassDeclaration = [{
    // Returns the tile sizes for all levels set for the op.
    TileSizesListType getTileSizeVals() const;

    // Returns the tile sizes for a level set for the op.
    SmallVector<int64_t> getTileSizeVals(unsigned level) const;

    // Returns the scalable tile flags for all levels set for the op.
    ScalableTileFlagsListType getScalableTileFlagVals();

    // Returns the scalable tile flags for a level set for the op.
    SmallVector<bool> getScalableTileFlagVals(unsigned level);

    // Returns the tile interchange for a level set for the op.
    SmallVector<int64_t> getTileInterchangeVals(unsigned level) const;

    // Returns true if there are no tile interchange values (this means that
    // interchange can be ignored).
    bool isInterchangeEmpty();

    // Returns the native vector size to use.
    SmallVector<int64_t> getNativeVectorSizeVals() {
      return SmallVector<int64_t>(getNativeVectorSize());
    }
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// tts_codegen.compilation_info
//===---------------------------------------------------------------------===//

def IREECodegen_CompilationInfoAttr :
    AttrDef<IREECodegen_Dialect, "CompilationInfo", []> {
  let mnemonic = "compilation_info";
  let summary = [{Drive lowering of an operation from input dialect}];
  let description = [{
    Specifies the information that allows controlling the compilation
    of operations like matmul/conv within Triton. This information is used to 
    override the defaults used by the compiler. If set on the input to the 
    compiler, there is no guarantee that the config survives until codegen. 
    Named operations like matmul/conv are more likely to retain their lowering
    configurations.

    TODO: It is expected that the `TranslationInfoAttr` and the
    `LoweringConfigAttr` are specified. Currently there is no
    verification that the values of the `LoweringConfigAttr` fully
    specifies the behaviour of the compilation path chosen with
    `TranslationInfoAttr`. This could be added in the future.
  }];
  let parameters = (ins
    AttrParameter<"LoweringConfigAttrInterface", "">:$loweringConfig,
    AttrParameter<"TranslationInfoAttr", "">:$translationInfo
  );

  let assemblyFormat = [{
    `<` `lowering_config` `=` $loweringConfig `,` `translation_info` `=` $translationInfo `>`
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// tts_codegen.export_config
//===---------------------------------------------------------------------===//

def IREECodegen_ExportConfig : AttrDef<IREECodegen_Dialect, "ExportConfig", []> {
  let mnemonic = "export_config";
  let summary = "User defined workgroup size specification";
  let description = [{
    Allows setting workgroup size for pre-formed dispatches.
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t", "Workgroup Size to use">:$workgroup_size
  );
  let assemblyFormat = [{
    `<` `workgroup_size` `=` `[` $workgroup_size `]` `>`
  }];
  let extraClassDeclaration = [{
    ArrayAttr getWorkgroupSizeIndexArray();
  }];
  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// tts_codegen.encoding_layout
//===---------------------------------------------------------------------===//

def IREECodegen_EncodingNopLayoutAttr  :
    AttrDef<IREECodegen_Dialect, "EncodingNopLayout", [
    DeclareAttrInterfaceMethods<IREECodegen_LayoutAttrInterface, [
        "getEncodingInfo",
        "lowerOp"
      ]>
    ]> {
  let mnemonic = "encoding_nop_layout";
  let summary = "An attribute with implementation that treats encoding as nop.";
  let description = [{
    An attribute that implements the interface methods that discards the
    encodings. It can be a default attribute when a backend does not implement
    encoding details.
  }];
}

#endif // IREE_COMP
