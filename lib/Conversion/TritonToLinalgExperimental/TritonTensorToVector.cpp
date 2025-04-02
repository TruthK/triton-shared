#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTENSORTOVECTOR
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// 转换 TTS_TransferReadOp 到 vector.transfer_read
class TransferReadOpConversion
    : public OpConversionPattern<tts::TransferReadOp> {
public:
  using OpConversionPattern<tts::TransferReadOp>::OpConversionPattern;
  TransferReadOpConversion(const TypeConverter &typeConverter,
                           MLIRContext *context)
      : OpConversionPattern<tts::TransferReadOp>(typeConverter, context) {}
  LogicalResult
  matchAndRewrite(tts::TransferReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value source = adaptor.getBase();
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto vectorType =
        VectorType::get(resultType.getShape(), resultType.getElementType());

    // 创建索引数组
    SmallVector<Value> indices(
        resultType.getRank(),
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0));

    // 创建 AffineMapAttr
    auto map = AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
        resultType.getRank(), rewriter.getContext()));

    // 处理 padding/other
    Value padding;
    if (op.getOther()) {
      padding = adaptor.getOther();
    } else {
      if (auto floatType =
              mlir::dyn_cast<FloatType>(resultType.getElementType())) {
        padding = rewriter.create<arith::ConstantFloatOp>(
            op.getLoc(), APFloat(0.0), floatType);
      } else if (auto intType =
                     mlir::dyn_cast<IntegerType>(resultType.getElementType())) {
        padding =
            rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, intType);
      }
    }

    // 处理 mask
    Value mask;
    if (op.hasMask()) {
      auto resultRank = resultType.getRank();
      auto maskDims = op.getMixedMaskDims();
      
      // 创建完整的维度数组
      SmallVector<Value> maskSizes;
      int maskDimIdx = 0;
      
      // 遍历所有维度
      for (int i = 0; i < resultRank; ++i) {
        if (maskDimIdx < maskDims.size()) {
          // 如果有对应的 mask 维度，使用它
          auto maskDim = maskDims[maskDimIdx];
          if (auto attr = maskDim.dyn_cast<Attribute>()) {
            // 处理静态维度
            if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
              maskSizes.push_back(rewriter.create<arith::ConstantIndexOp>(
                  op.getLoc(), intAttr.getInt()));
            }
          } else {
            // 处理动态维度
            maskSizes.push_back(cast<Value>(maskDim));
          }
          maskDimIdx++;
        } else {
          // 如果没有对应的 mask 维度，使用结果类型的对应维度
          maskSizes.push_back(rewriter.create<arith::ConstantIndexOp>(
              op.getLoc(), resultType.getDimSize(i)));
        }
      }

      // 创建向量掩码
      mask = rewriter.create<vector::CreateMaskOp>(
          op.getLoc(), 
          VectorType::get(resultType.getShape(), rewriter.getI1Type()),
          maskSizes);
    }

    // 创建 inBounds 属性
    SmallVector<bool> inBoundsValues(resultType.getRank(), true);
    auto inBounds = rewriter.getBoolArrayAttr(inBoundsValues);

    // 创建 transfer_read
    auto xferRead = rewriter.create<vector::TransferReadOp>(
        op.getLoc(),
        vectorType, // 结果类型
        source,     // 源内存引用
        indices,    // 索引
        map,        // permutation map
        padding,    // padding 值
        mask,       // mask (可选)
        inBounds    // inBounds
    );
    rewriter.replaceOp(op, xferRead);
    return success();
  }
};

// 类型转换器：将 tensor 类型转换为 vector 类型
class TensorToVectorTypeConverter : public TypeConverter {
public:
  TensorToVectorTypeConverter() {
    // 添加默认转换规则
    addConversion([](Type type) { return type; });

    // 添加 tensor 到 vector 的转换规则
    addConversion([](RankedTensorType tensorType) -> Type {
      tensorType.dump();
      return VectorType::get(tensorType.getShape(),
                             tensorType.getElementType());
    });

    // 添加 materialization 转换
    auto materializeCast = [](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
      inputs[0].dump();
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };

    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
    addArgumentMaterialization(materializeCast);
  }
};

class TritonTensorToVectorPass
    : public triton::impl::TritonTensorToVectorBase<TritonTensorToVectorPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, 
                   func::FuncDialect,
                   arith::ArithDialect,
                   scf::SCFDialect,
                   tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp moduleOp = getOperation();

    // 创建类型转换器
    TensorToVectorTypeConverter typeConverter;

    // 创建转换目标
    ConversionTarget target(*context);


    // 设置需要转换的操作
    target.addIllegalOp<tts::TransferReadOp>();
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      return !isa<RankedTensorType>(op.getType());
    });


    // 创建重写模式集
    RewritePatternSet patterns(context);

    // 添加转换模式
    patterns.add<TransferReadOpConversion>(context);

    // 添加函数签名转换模式
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // 应用转换
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // 添加清理 pass
    PassManager pm(context, moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, moduleOp))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonTensorToVectorPass() {
  return std::make_unique<TritonTensorToVectorPass>();
}