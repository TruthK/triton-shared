//===- PtrTransform.cpp - Ptr transformation pass
//--------------------------===//
//
// Part of the Triton-Shared project under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to transform function signatures for
// matmul_kernel functions by converting !llvm.ptr arguments to !llvm.ptr<1> and
// removing associated i64 arguments.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "ptr-transform"

namespace mlir::tts {
#define GEN_PASS_DEF_PTRTRANSFORMPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct PtrTransformPass : impl::PtrTransformPassBase<PtrTransformPass> {

}; // struct PtrTransformPass

} // anonymous namespace

} // namespace mlir::tts