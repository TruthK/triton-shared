//===- PtrTransform.cpp - Ptr transformation pass ------------------------===//

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file implements a pass to transform function signatures for
// matmul_kernel functions by converting !llvm.ptr arguments to !llvm.ptr<1> and
// removing associated i64 arguments.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h" // for LLVMPointerType
#include "mlir/IR/Block.h"                 // For Block, BlockArgument
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h" // For ModuleOp
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h" // llvm::seq, llvm::equal
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeSupport.h" // for mlir::isa
#include "llvm/ADT/APInt.h"

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
    registry.insert<mlir::LLVM::LLVMDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    transformPointerParams(module);
    removeUnusedI64Args(module);
    adjustGetElementPtrParams(module);
    removeLoadAfterGEP(module);
  }

private:
  void transformPointerParams(ModuleOp module);
  void removeUnusedI64Args(ModuleOp module);
  void adjustGetElementPtrParams(ModuleOp module);
  void removeLoadAfterGEP(ModuleOp module);
}; // struct PtrTransformPass

// 将所有 pointer 参数转换为 address space 1，保留最后一个参数不变
void PtrTransformPass::transformPointerParams(ModuleOp module) {
  auto ctx = module.getContext();
  module.walk([&](LLVM::LLVMFuncOp func) {
    auto oldType = func.getFunctionType();
    int numParams = oldType.getNumParams();
    if (numParams <= 1)
      return;
    SmallVector<Type> newParams;
    newParams.reserve(numParams);
    for (int i = 0; i < numParams - 1; ++i) {
      Type param = oldType.getParamType(i);
      if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(param))
        newParams.push_back(LLVM::LLVMPointerType::get(ctx, 1));
      else
        newParams.push_back(param);
    }
    newParams.push_back(oldType.getParamType(numParams - 1));
    if (!llvm::equal(newParams, oldType.getParams())) {
      auto newType = LLVM::LLVMFunctionType::get(
          oldType.getReturnType(), newParams, oldType.isVarArg());
      function_interface_impl::setFunctionType(func, newType);
      Block &entry = func.getBody().front();
      for (unsigned i = 0, e = newParams.size(); i < e; ++i)
        entry.getArgument(i).setType(newParams[i]);
    }
  });
}

// 删除未使用的 i64 参数
void PtrTransformPass::removeUnusedI64Args(ModuleOp module) {
  module.walk([&](LLVM::LLVMFuncOp func) {
    int numArgs = func.getNumArguments();
    if (numArgs <= 1)
      return;
    llvm::BitVector toErase(numArgs);
    Block &entry = func.getBody().front();
    for (int i = 0; i < numArgs - 1; ++i)
      if (entry.getArgument(i).use_empty())
        toErase.set(i);
    if (!toErase.any())
      return;
    auto oldType = func.getFunctionType();
    SmallVector<Type> keptParams;
    keptParams.reserve(oldType.getNumParams());
    for (int i = 0, e = oldType.getNumParams(); i < e; ++i)
      if (!toErase.test(i))
        keptParams.push_back(oldType.getParamType(i));
    auto newType = LLVM::LLVMFunctionType::get(
        oldType.getReturnType(), keptParams, oldType.isVarArg());
    function_interface_impl::eraseFunctionArguments(func, toErase, newType);
  });
}

// 将 llvm.getelementptr 操作中所有索引常数 1 修改为 0，仅针对基于函数参数的 GEP
void PtrTransformPass::adjustGetElementPtrParams(ModuleOp module) {
  module.walk([&](LLVM::GEPOp gep) {
    Value basePtr = gep.getBase();
    // 仅处理基于函数参数（BlockArgument）的 GEP
    if (!mlir::isa<mlir::BlockArgument>(basePtr))
      return;
    // 获取所有常量索引
    auto rawConst = gep.getRawConstantIndices();
    if (rawConst.empty())
      return;
    // 构造可修改的副本并替换索引 1 为 0
    SmallVector<int32_t, 4> newRaw(rawConst.begin(), rawConst.end());
    bool changed = false;
    for (auto &idx : newRaw) {
      if (idx == 1) {
        idx = 0;
        changed = true;
      }
    }
    if (changed)
      gep.setRawConstantIndices(newRaw);
  });
}

// 删除由函数参数作为基址的 GEP 后紧跟的 LoadOp，并用 GEP 的结果替换 LoadOp 的结果
void PtrTransformPass::removeLoadAfterGEP(ModuleOp module) {
  module.walk([&](LLVM::LoadOp loadOp) {
    auto addr = loadOp.getAddr();
    // 检查 Load 的地址是否由 GEP 生成
    if (auto gepOp = addr.getDefiningOp<LLVM::GEPOp>()) {
      auto base = gepOp.getBase();
      // 仅处理基于函数参数的 GEP，且参数类型为 LLVM 指针
      if (!mlir::isa<mlir::BlockArgument>(base)) return;
      if (!mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) return;
      // 用 GEP 的结果替换所有 LoadOp 的使用
      loadOp.replaceAllUsesWith(gepOp.getResult());
      // 删除 LoadOp 操作
      loadOp.getOperation()->erase();
    }
  });
}

} // namespace
} // namespace mlir::tts