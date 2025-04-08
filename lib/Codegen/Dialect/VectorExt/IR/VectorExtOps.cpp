// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <utility>
using namespace mlir;
using namespace mlir::tts::IREE::VectorExt;

using VectorValue = TypedValue<VectorType>;

//===----------------------------------------------------------------------===//
// LayoutConflictResolutionOp
//===----------------------------------------------------------------------===//

// Validate that the layout has the same shape as the input.
LogicalResult ToLayoutOp::verify() {
  return getLayout().isValidLayout(getInput().getType(), getLoc());
}

// to_simd -> to_simt
OpFoldResult ToSIMDOp::fold(FoldAdaptor) {
  if (auto simtOp = getOperand().getDefiningOp<ToSIMTOp>()) {
    return simtOp.getOperand();
  }
  return {};
}

// to_simt -> to_simd
OpFoldResult ToSIMTOp::fold(FoldAdaptor) {
  if (auto simdOp = getOperand().getDefiningOp<ToSIMDOp>()) {
    return simdOp.getOperand();
  }
  return {};
}

// clang-format off
#define GET_OP_CLASSES
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format on

//===----------------------------------------------------------------------===//
// TransferReadOp
//===----------------------------------------------------------------------===//

void TransferReadOp::build(OpBuilder &b, OperationState &state, Type resultType,
                           Value base, ArrayRef<OpFoldResult> indices,
                           ArrayRef<OpFoldResult> mask_dims, Value other) {
  // 将 OpFoldResult 数组转换为 ValueRange
  SmallVector<Value> dynamicIndices;
  SmallVector<int64_t> staticIndices;
  dispatchIndexOpFoldResults(indices, dynamicIndices, staticIndices);

  // 处理 mask_dims
  SmallVector<Value> dynamicMaskDims;
  SmallVector<int64_t> staticMaskDims;
  dispatchIndexOpFoldResults(mask_dims, dynamicMaskDims, staticMaskDims);

  build(b, state, resultType, base, dynamicIndices,
        b.getDenseI64ArrayAttr(staticIndices), dynamicMaskDims,
        b.getDenseI64ArrayAttr(staticMaskDims), other);
}

//===----------------------------------------------------------------------===//
// TransferWriteOp
//===----------------------------------------------------------------------===//

void TransferWriteOp::build(OpBuilder &b, OperationState &state, Value base,
                            Value value, ArrayRef<OpFoldResult> indices,
                            ArrayRef<OpFoldResult> mask_dims) {
  // 将 OpFoldResult 数组转换为 ValueRange
  SmallVector<Value> dynamicIndices;
  SmallVector<int64_t> staticIndices;
  dispatchIndexOpFoldResults(indices, dynamicIndices, staticIndices);

  // 处理 mask_dims
  SmallVector<Value> dynamicMaskDims;
  SmallVector<int64_t> staticMaskDims;
  dispatchIndexOpFoldResults(mask_dims, dynamicMaskDims, staticMaskDims);

  build(b, state, base, value, dynamicIndices,
        b.getDenseI64ArrayAttr(staticIndices), dynamicMaskDims,
        b.getDenseI64ArrayAttr(staticMaskDims));
}
