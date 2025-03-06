#ifndef TTS_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES
#define TTS_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES

include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenDialect.td"

def IREECodegen_LoweringConfigAttrInterface :
  AttrInterface<"LoweringConfigAttrInterface"> {
  let cppNamespace = "::mlir::tts::GPU";
  let description = [{
    Attribute interface for specifying information used by the compiler to
    translate a specific operation. The way this information is used is backend
    and/or pipeline specific, so this interface only exposes information needed
    for shared use cases.
  }];

  let methods = [
    InterfaceMethod<
      [{
        Returns the tile sizes to use for workgroup distribution.

        TODO: This should be queried/constructed from device information and
        used to compute workgroup size.
      }],
      "::llvm::SmallVector<int64_t>",
      "getWorkgroupTileSizes",
      (ins),
      "",
      [{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      [{
        Returns the loop interchange to use for workgroup distribution.
      }],
      "::llvm::SmallVector<int64_t>",
      "getWorkgroupInterchange",
      (ins),
      "",
      [{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      [{
        Returns true if the lowering config specifies tile sizes for the given
        tiling level.
      }],
      "bool",
      "hasTilingLevel",
      (ins "unsigned":$level),
      "",
      [{
        return false;
      }]
    >,
    InterfaceMethod<
      [{
        Returns true if the lowering config specifies tile sizes for the
        workgroup tiling level.
      }],
      "bool",
      "hasWorkgroupTilingLevel",
      (ins),
      "",
      [{
        return false;
      }]
    >,
    InterfaceMethod<
      [{
        Returns the tile sizes for the specified tiling level. The
        interpretation of |level| is attribute and backend dependent. The
        |target| is the operation this lowering configuration annotates.

        returns an empty list if sizes are not specified for this level. dynamic
        sizes are specified with `shapedtype::kdynamicsize`.
      }],
      "::llvm::SmallVector<int64_t>",
      "getStaticTilingLevelSizes",
      (ins
        "unsigned":$level,
        "::mlir::Operation *":$target
      ),
      "",
      [{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      [{
        Constructs the tile sizes for the specified level. The
        interpretation of |level| is attribute and backend dependent. The
        |target| is the operation this lowering configuration annotates.

        Returns an empty list if sizes are not specified for this level.
      }],
      "::llvm::SmallVector<::mlir::OpFoldResult>",
      "getTilingLevelSizes",
      (ins
        "::mlir::OpBuilder &":$builder,
        "unsigned":$level,
        "::mlir::Operation *":$target
      ),
      "",
      [{
        return ::llvm::SmallVector<OpFoldResult>();
      }]
    >
  ];
}

def IREECodegen_LayoutAttrInterface :
  AttrInterface<"LayoutAttrInterface"> {
  let cppNamespace = "::mlir::tts::GPU";
  let description = [{
    An interface that collects a set of methods for encoding materialization.
  }];

  let methods = [
    InterfaceMethod<
      [{
        Returns the layout of materialized encoding for a tensor type.
      }],
      "::mlir::tts::GPU::MaterializeEncodingInfo",
      "getEncodingInfo",
      (ins "::mlir::RankedTensorType":$type),
      "",
      [{
        assert(false && "unimplemented interface method");
        return MaterializeEncodingInfo{};
      }]
    >,
    InterfaceMethod<
      [{
        Returns the layout of materialized encoding for a tensor type.
      }],
      "::mlir::Operation *",
      "lowerOp",
      (ins "::mlir::OpBuilder &":$b,
                    "::mlir::Operation *":$op,
                    "::mlir::TypeRange":$convertedResTypes,
                    "::mlir::ValueRange":$convertedOperands),
      "",
      [{
        assert(false && "unimplemented interface method");
        return nullptr;
      }]
    >
  ];
}

#endif // TTS_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES
