//===- PtrTransform.cpp - Ptr transformation pass ------------------------===//

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file implements a pass to transform function signatures for
// matmul_kernel functions by converting !llvm.ptr arguments to !llvm.ptr<1> and
// removing associated i64 arguments.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"        // For ModuleOp
#include "mlir/IR/Block.h"             // For Block, BlockArgument
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"  // llvm::seq
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ptr-transform"

namespace mlir::tts {
// 使用 tablegen 生成的 Pass 定义
#define GEN_PASS_DEF_PTRTRANSFORMPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct PtrTransformPass : impl::PtrTransformPassBase<PtrTransformPass> {
public:
  using impl::PtrTransformPassBase<PtrTransformPass>::PtrTransformPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, 
                    IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](LLVM::LLVMFuncOp func) {
      int numArgs = func.getNumArguments();
      if (numArgs <= 1)
        return;
      llvm::BitVector toErase(numArgs);
      Block &entry = func.getBody().front();
      for (int i = 0; i < numArgs - 1; ++i) {
        BlockArgument arg = entry.getArgument(i);
        if (arg.use_empty())
          toErase.set(i);
      }
      if (toErase.any()) {
        auto oldType = func.getFunctionType();
        SmallVector<Type> newArgTypes;
        newArgTypes.reserve(oldType.getNumParams());
        for (int i = 0, e = oldType.getNumParams(); i < e; ++i)
          if (!toErase.test(i))
            newArgTypes.push_back(oldType.getParamType(i));
        auto newType = LLVM::LLVMFunctionType::get(
            oldType.getReturnType(), newArgTypes, oldType.isVarArg());
        function_interface_impl::eraseFunctionArguments(func, toErase, newType);
      }
    });
  }
}; // struct PtrTransformPass

} // namespace
} // namespace mlir::tts