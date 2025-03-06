#ifndef TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUINTERFACES_H_
#define TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUINTERFACES_H_

include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUDialect.td"
include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUEnums.td"
include "mlir/IR/EnumAttr.td"

def TTSGPU_MmaInterfaceAttr : AttrInterface<"MmaInterfaceAttr"> {
  let cppNamespace = "::mlir::tts::GPU";

  let description = [{
    Interface used to query information needed to generate code targeting a
    specific Matrix Multiply-Accumulate (MMA) machine intrinsic instruction.
    Layout information can be explicit or opaque, however all such attributes
    must specify the static shape of the operation and the required element
    types.
  }];

  let methods = [
    InterfaceMethod<
      [{
        Returns the shape of the MMA operation:
          C += A * B
        Where, up to some transpositions, `A` has shape `<M, K>`, `B` has shape
        `<K, N>`, and `C` has shape `<M, N>`.
      }],
      "::std::tuple<int64_t, int64_t, int64_t>",
      "getMNKShape",
      (ins)
    >,
    InterfaceMethod<
      [{
        Returns the element types of the operands to the MMA operation:
          C += A * B
      }],
      "::std::tuple<::mlir::Type, ::mlir::Type, ::mlir::Type>",
      "getABCElementTypes",
      (ins)
    >,
    InterfaceMethod<
      [{
        Returns the vector types of the operands to the MMA operation:
          C += A * B
        Each thread participating in the MMA operation is responsible for
        storing such shaped vectors in the registers.
      }],
      "::std::tuple<::mlir::VectorType, ::mlir::VectorType, ::mlir::VectorType>",
      "getABCVectorTypes",
      (ins)
    >,
    InterfaceMethod<
      [{
        Returns the vector layout for the A/B/C matrix of the MMA operation:
          C += A * B
      }],
      "::mlir::FailureOr<::std::tuple<"
        "::mlir::tts::VectorExt::VectorLayoutInterface, "
        "::mlir::tts::VectorExt::VectorLayoutInterface, "
        "::mlir::tts::VectorExt::VectorLayoutInterface>>",
      "getContractionLayout",
      (ins "::mlir::vector::ContractionOp":$contract),
      "",
      [{ return failure(); }]
    >,
    InterfaceMethod<
      [{
        Returns the expected subgroup size of the MMA operation/intrinsic.
      }],
      "int64_t",
      "getSubgroupSize",
      (ins)
    >,
    InterfaceMethod<
      [{
        Constructs a concrete instance of the associated mma operation.
      }],
      "::mlir::FailureOr<::mlir::Value>",
      "buildMmaOperation",
      (ins
        "::mlir::OpBuilder&":$builder,
        "::mlir::Location":$loc,
        "::mlir::Type":$result_type,
        "::mlir::Value":$lhs,
        "::mlir::Value":$rhs,
        "::mlir::Value":$acc
      ),
      "",
      [{ return failure(); }]
    >,
    InterfaceMethod<
      [{
        Constructs the offsets/sizes/strides for extracting the per-thread
        slice of the given operand fragment.
      }],
      "::mlir::LogicalResult",
      "populateOperandOffsetsSizesStrides",
      (ins
        "::mlir::OpBuilder&":$builder,
        "::mlir::Location":$loc,
        "::mlir::tts::GPU::MMAFragment":$fragment,
        "::mlir::Value":$lane_id,
        "::llvm::ArrayRef<int64_t>":$permutation,
        "::llvm::SmallVector<::mlir::OpFoldResult>&":$offsets,
        "::llvm::SmallVector<::mlir::OpFoldResult>&":$sizes,
        "::llvm::SmallVector<::mlir::OpFoldResult>&":$strides
      ),
      "",
      [{ return failure(); }]
    >,
    InterfaceMethod<
      [{
        Populates the reassociation indices and result shape to materialize the
        thread layout shape from the subgroup block shape.
      }],
      "::mlir::LogicalResult",
      "materializeOperandConcreteShape",
      (ins
        "::mlir::OpBuilder&":$builder,
        "::mlir::tts::GPU::MMAFragment":$fragment,
        "::mlir::Value":$operand,
        "std::optional<::llvm::ArrayRef<int64_t>>":$permutation,
        "::llvm::SmallVector<::mlir::SmallVector<int64_t, 2>>&":$reassociations,
        "::mlir::RankedTensorType&":$result_type
      ),
      "",
      [{ return failure(); }]
    >,
    InterfaceMethod<
      [{
        Returns the scope of the MMA operation. See TTSGPUEnums.td for
        available options.
      }],
      "::mlir::FailureOr<::mlir::tts::GPU::MMAScope>",
      "getMmaScope",
      (ins),
      "",
      [{ return failure(); }]
    >,
  ];
}

def TTSGPU_AnyMmaAttr : Attr<Or<[
  CPred<"isa<tts::GPU::MmaInterfaceAttr>($_self)">,
]>, "buffer-like constant attribute values"> {
  let storageType = [{ tts::GPU::MmaInterfaceAttr }];
  let returnType = [{ tts::GPU::MmaInterfaceAttr }];
  let convertFromStorage = "$_self";
}

#endif // TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUINTERFACES_H_
