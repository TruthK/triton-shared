//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_LINALGGENERICFUSIONPASS
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

} // namespace triton
} // namespace mlir

namespace {

// 判断两个linalg.generic操作是否可以融合
bool areGenericsFusable(linalg::GenericOp producer,
                        linalg::GenericOp consumer) {
  // 检查producer的输出是否是consumer的输入
  bool hasDataDependence = false;
  for (OpOperand &consumerOperand : consumer->getOpOperands()) {
    if (consumerOperand.get().getDefiningOp() == producer) {
      hasDataDependence = true;
      break;
    }
  }
  if (!hasDataDependence)
    return false;

  // 检查迭代空间是否相同
  if (producer.getNumLoops() != consumer.getNumLoops())
    return false;

  auto producerIteratorTypes = producer.getIteratorTypesArray();
  auto consumerIteratorTypes = consumer.getIteratorTypesArray();
  if (producerIteratorTypes.size() != consumerIteratorTypes.size())
    return false;

  for (size_t i = 0; i < producerIteratorTypes.size(); ++i) {
    if (producerIteratorTypes[i] != consumerIteratorTypes[i])
      return false;
  }

  // 可以添加更多条件判断融合的可行性
  return true;
}

// 融合两个linalg.generic操作的pattern
struct FuseLinalgGenericsPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumerOp,
                                PatternRewriter &rewriter) const override {
    if (isa<linalg::MatmulOp>(consumerOp))
      return failure();
    // 寻找可能的producer
    linalg::GenericOp producerOp = nullptr;
    int64_t producerIdx = -1;
    for (OpOperand &operand : consumerOp->getOpOperands()) {
      if (auto defOp = operand.get().getDefiningOp<linalg::GenericOp>()) {
        if (areGenericsFusable(defOp, consumerOp)) {
          producerOp = defOp;
          producerIdx = operand.getOperandNumber();
          break;
        }
      }
    }

    if (!producerOp ||isa<linalg::MatmulOp>(producerOp))
      return failure();

    // 创建新的融合后的linalg.generic操作
    // 这是一个简化版的实现，实际情况需要处理更复杂的场景
    Location loc = consumerOp.getLoc();

    // 收集所有输入和输出
    SmallVector<Value, 4> fusedInputs;
    SmallVector<Value, 4> fusedOutputs;
    SmallVector<AffineMap, 4> fusedMaps;

    // 添加producer的输入
    for (auto input : producerOp.getDpsInputOperands()) {
      fusedInputs.push_back(input->get());
      size_t idx = input->getOperandNumber();
      fusedMaps.push_back(producerOp.getIndexingMapsArray()[idx]);
    }

    // 添加consumer的输入，除了与producer相连的那个
    for (auto input : consumerOp.getDpsInputOperands()) {
      if (input->getOperandNumber() != producerIdx) {
        fusedInputs.push_back(input->get());
        size_t idx = input->getOperandNumber();
        fusedMaps.push_back(consumerOp.getIndexingMapsArray()[idx]);
      }
    }

    // 输出使用consumer的输出
    fusedOutputs = consumerOp.getOutputs();
    size_t numInputs = consumerOp.getInputs().size();
    for (size_t i = 0; i < consumerOp.getOutputs().size(); ++i) {
      fusedMaps.push_back(consumerOp.getIndexingMapsArray()[numInputs + i]);
    }

    // 创建迭代器类型
    SmallVector<utils::IteratorType> iteratorTypes;
    // 假设所有迭代器都是parallel类型，这在大多数情况下是正确的
    // 如果需要更精确的处理，需要根据实际情况分析
    for (size_t i = 0; i < consumerOp.getIteratorTypesArray().size(); ++i) {
      iteratorTypes.push_back(utils::IteratorType::parallel);
    }

    auto fusedGeneric = rewriter.create<linalg::GenericOp>(
        loc, consumerOp.getResultTypes(), fusedInputs, fusedOutputs, fusedMaps,
        iteratorTypes,
        /*bodyBuilder=*/[&](OpBuilder &builder, Location loc, ValueRange args) {
          // 分割参数，找到对应于producer输入/输出和consumer输入/输出的部分
          size_t producerInputCount = producerOp.getNumDpsInputs();
          size_t consumerInputCount = consumerOp.getNumDpsInputs() - 1; // 减1是因为我们不包括producer的输出
          size_t totalInputCount = producerInputCount + consumerInputCount;
          
          // 获取producer的输入
          SmallVector<Value> producerInputs;
          for (size_t i = 0; i < producerInputCount; ++i) {
            producerInputs.push_back(args[i]);
          }
          
          // 获取consumer的输入（不包括来自producer的那个）
          SmallVector<Value> consumerInputs;
          for (size_t i = 0; i < consumerInputCount; ++i) {
            consumerInputs.push_back(args[producerInputCount + i]);
          }
          
          // 获取输出
          SmallVector<Value> outputs;
          for (size_t i = totalInputCount; i < args.size(); ++i) {
            outputs.push_back(args[i]);
          }
          
          // 克隆producer的计算逻辑
          Region &producerRegion = producerOp.getRegion();
          Block &producerBlock = producerRegion.front();
          
          // 将producer的输入映射到参数
          IRMapping producerMapping;
          for (size_t i = 0; i < producerInputCount; ++i) {
            producerMapping.map(producerBlock.getArgument(i), producerInputs[i]);
          }
          // 输出位置先不映射，我们需要获取计算结果
          
          // 克隆除了yield以外的所有操作
          for (auto &op : producerBlock.getOperations()) {
            if (!isa<linalg::YieldOp>(op)) {
              builder.clone(op, producerMapping);
            }
          }
          
          // 获取producer yield操作的操作数，这是中间结果
          auto producerYieldOp = cast<linalg::YieldOp>(producerBlock.getTerminator());
          Value producerResult = producerMapping.lookupOrDefault(producerYieldOp.getOperand(0));
          
          // 克隆consumer的计算逻辑
          Region &consumerRegion = consumerOp.getRegion();
          Block &consumerBlock = consumerRegion.front();
          
          // producerIdx已在匹配阶段确定，无需重新查找
          
          // 将consumer的输入映射到参数，将producer的结果映射到对应位置
          IRMapping consumerMapping;
          int consumerInputIdx = 0;
          for (size_t i = 0; i < consumerOp.getNumDpsInputs(); ++i) {
            if (i == producerIdx) {
              // 将producer的输出映射到consumer对应的输入位置
              consumerMapping.map(consumerBlock.getArgument(i), producerResult);
            } else {
              // 将其他输入正常映射
              consumerMapping.map(consumerBlock.getArgument(i), consumerInputs[consumerInputIdx++]);
            }
          }
          
          // 将输出映射到对应位置
          for (size_t i = 0; i < outputs.size(); ++i) {
            size_t outputIdx = consumerOp.getNumDpsInputs() + i;
            consumerMapping.map(consumerBlock.getArgument(outputIdx), outputs[i]);
          }
          
          // 克隆除了yield以外的所有操作
          for (auto &op : consumerBlock.getOperations()) {
            if (!isa<linalg::YieldOp>(op)) {
              builder.clone(op, consumerMapping);
            }
          }
          
          // 获取consumer yield操作的操作数，这是最终结果
          auto consumerYieldOp = cast<linalg::YieldOp>(consumerBlock.getTerminator());
          Value result = consumerMapping.lookupOrDefault(consumerYieldOp.getOperand(0));
          
          // 创建最终的yield操作
          builder.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(consumerOp, fusedGeneric.getResults());
    return success();
  }
};

// 实现Pass
struct LinalgGenericFusionPass
    : public triton::impl::LinalgGenericFusionPassBase<
          LinalgGenericFusionPass> {
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      RewritePatternSet patterns(context);
      patterns.add<FuseLinalgGenericsPattern>(context);

      // 为每个函数应用融合模式
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {

// 实现创建pass的工厂函数
std::unique_ptr<OperationPass<ModuleOp>> createLinalgGenericFusionPass() {
  return std::make_unique<LinalgGenericFusionPass>();
}

} // namespace triton
} // namespace mlir