//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGEXPERIMENTAL_H
#define TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGEXPERIMENTAL_H

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgExperimentalPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonStructuredToVectorPass();

std::unique_ptr<OperationPass<ModuleOp>> createTritonTensorToVectorPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGEXPERIMENTAL_H
