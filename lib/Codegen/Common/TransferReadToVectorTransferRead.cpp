// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tts;
using namespace mlir::tts::IREE::VectorExt;

namespace mlir::tts {

#define GEN_PASS_DEF_VECTOREXTTRANSFERTOVECTORTRANSFERPASS
#include "triton-shared/Codegen/Common/Passes.h.inc"

namespace {

/// 将 iree_vector_ext.transfer_read 转换为 vector.transfer_read
struct ConvertVectorExtTransferReadToVectorTransferRead
    : public OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferReadOp op,
                                PatternRewriter &rewriter) const override {

    /**在include/triton-shared/Codegen/Common/Passes.td中定义，在lib/Codegen/Common下新建一个文件来实现一个pass，pass中有pattern，pattern中将iree_vector_ext.transfer_read
     * lowering 到vector.transfer_read,iree_vector_ext.transfer_read的use是memref.copy,才可以执行lowering，如果不是则报错。
     * ，loweering的时候vector.transfer_read的ptr就是iree_vector_ext.transfer_read，indies也是iree_vector_ext.transfer_read的，但是mask的算法是：max(
     * mask[dim1]% 结果类型的形状[dim1] , 结果类型的形状[dim1]
     * ），并且vector.transfer_read的结果类型是vector */
    return success();
  }
};

struct VectorExtTransferToVectorTransferPass
    : impl::VectorExtTransferToVectorTransferPassBase<
          VectorExtTransferToVectorTransferPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加转换模式
    patterns.add<ConvertVectorExtTransferReadToVectorTransferRead>(context);

    // 应用模式
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

} // namespace mlir::tts