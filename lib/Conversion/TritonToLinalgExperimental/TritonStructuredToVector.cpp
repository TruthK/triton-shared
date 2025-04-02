#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

using namespace mlir;
using namespace mlir::tts;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONSTRUCTUREDTOVECTOR
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

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

    // 创建 AffineMapAttr (注意这里的改变)
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
      mask.dump();
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
        map,        // permutation map (现在是 AffineMapAttr)
        padding,    // padding 值
        mask,       // mask (可选)
        inBounds    // inBounds (现在是 ArrayAttr)
    );
    rewriter.replaceOp(op, xferRead);
    return success();
  }
};

// 转换 TTS_TransferWriteOp 到 vector.transfer_write
class TransferWriteOpConversion
    : public OpConversionPattern<tts::TransferWriteOp> {
public:
  using OpConversionPattern<tts::TransferWriteOp>::OpConversionPattern;
  TransferWriteOpConversion(const TypeConverter &typeConverter,
                            MLIRContext *context)
      : OpConversionPattern<tts::TransferWriteOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::TransferWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dest = adaptor.getBase();
    Value value = adaptor.getValue();

    // 如果value是tensor类型,转换为vector
    if (isa<RankedTensorType>(value.getType())) {
      auto tensorType = cast<RankedTensorType>(value.getType());
      auto vectorType =
          VectorType::get(tensorType.getShape(), tensorType.getElementType());
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          op.getLoc(), vectorType, value);
      value = cast->getResult(0);
    }

    auto valueType = cast<VectorType>(value.getType());

    // 创建索引数组
    SmallVector<Value> indices(
        valueType.getRank(),
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0));

    // 创建 AffineMapAttr (注意这里的改变)
    auto map = AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
        valueType.getRank(), rewriter.getContext()));

    // 处理 mask
    Value mask;
    if (op.hasMask()) {
      auto valueRank = valueType.getRank();
      auto maskDims = op.getMixedMaskDims();

      // 创建完整的维度数组
      SmallVector<Value> maskSizes;
      int maskDimIdx = 0;

      // 遍历所有维度
      for (int i = 0; i < valueRank; ++i) {
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
          // 如果没有对应的 mask 维度，使用值类型的对应维度
          maskSizes.push_back(rewriter.create<arith::ConstantIndexOp>(
              op.getLoc(), valueType.getDimSize(i)));
        }
      }

      // 创建向量掩码
      mask = rewriter.create<vector::CreateMaskOp>(
          op.getLoc(),
          VectorType::get(valueType.getShape(), rewriter.getI1Type()),
          maskSizes);
    }

    // 创建 inBounds 属性
    SmallVector<bool> inBoundsValues(valueType.getRank(), true);
    auto inBounds = rewriter.getBoolArrayAttr(inBoundsValues);

    // 创建 transfer_write
    auto xferWrite = rewriter.create<vector::TransferWriteOp>(
        op.getLoc(),
        value,   // 向量值
        dest,    // 目标内存引用
        indices, // 索引
        map,     // permutation map (现在是 AffineMapAttr)
        mask,    // mask (可选)
        inBounds // inBounds (现在是 ArrayAttr)
    );

    rewriter.replaceOp(op, xferWrite);
    return success();
  }
};

template <typename TruncOp>
class ArithTypeConversionPattern : public OpConversionPattern<TruncOp> {
public:
  using OpConversionPattern<TruncOp>::OpConversionPattern;

  ArithTypeConversionPattern(const TypeConverter &typeConverter,
                             MLIRContext *context)
      : OpConversionPattern<TruncOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(TruncOp op, typename TruncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 只处理tensor类型的trunc
    auto inputType = op.getIn().getType();
    if (!mlir::isa<RankedTensorType>(inputType))
      return failure();

    auto tensorType = cast<RankedTensorType>(inputType);
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    // 创建对应的vector类型
    auto srcVectorType =
        VectorType::get(tensorType.getShape(), tensorType.getElementType());
    auto dstVectorType =
        VectorType::get(resultType.getShape(), resultType.getElementType());

    // 将输入tensor转换为vector
    auto inputCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), srcVectorType, adaptor.getIn());

    // 创建vector trunc操作
    auto truncOp = rewriter.create<TruncOp>(op.getLoc(), dstVectorType,
                                            inputCast.getResult(0));

    // // 将结果vector转换回tensor
    // auto resultCast = rewriter.create<UnrealizedConversionCastOp>(
    //     op.getLoc(), resultType, truncOp.getResult());

    rewriter.replaceOp(op, truncOp);
    return success();
  }
};

// Linalg MatMul 的转换模式
class MatmulOpConverter : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  MatmulOpConverter(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<linalg::MatmulOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 获取输入和输出值并转换为vector类型
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];

    // 确保所有操作数都是vector类型
    if (isa<RankedTensorType>(lhs.getType())) {
      auto tensorType = cast<RankedTensorType>(lhs.getType());
      auto vectorType = VectorType::get(tensorType.getShape(), 
                                      tensorType.getElementType());
      lhs = rewriter.create<UnrealizedConversionCastOp>(
          loc, vectorType, lhs).getResult(0);
    }

    if (isa<RankedTensorType>(rhs.getType())) {
      auto tensorType = cast<RankedTensorType>(rhs.getType());
      auto vectorType = VectorType::get(tensorType.getShape(), 
                                      tensorType.getElementType());
      rhs = rewriter.create<UnrealizedConversionCastOp>(
          loc, vectorType, rhs).getResult(0);
    }

    // 获取输出tensor类型并创建对应的vector类型
    auto outputTensorType = cast<RankedTensorType>(output.getType());
    auto outputVectorType = VectorType::get(outputTensorType.getShape(),
                                          outputTensorType.getElementType());

    if (isa<RankedTensorType>(output.getType())) {
      output = rewriter.create<UnrealizedConversionCastOp>(
          loc, outputVectorType, output).getResult(0);
    }

    // 创建 AffineMap 表示矩阵乘法的索引映射
    auto context = rewriter.getContext();
    AffineExpr m = rewriter.getAffineDimExpr(0);  // 矩阵 A 的行
    AffineExpr k = rewriter.getAffineDimExpr(1);  // 收缩维度
    AffineExpr n = rewriter.getAffineDimExpr(2);  // 矩阵 B 的列

    // 修改映射以正确表示矩阵乘法的维度关系
    // A[m,k] * B[k,n] = C[m,n]
    auto mapA = AffineMap::get(3, 0, {m, k}, context);      // (m, k, n) -> (m, k)
    auto mapB = AffineMap::get(3, 0, {k, n}, context);      // (m, k, n) -> (k, n)
    auto mapC = AffineMap::get(3, 0, {m, n}, context);      // (m, k, n) -> (m, n)

    SmallVector<Attribute> indexingMapsAttrs = {
      AffineMapAttr::get(mapA),
      AffineMapAttr::get(mapB),
      AffineMapAttr::get(mapC)
    };
    auto indexingMaps = rewriter.getArrayAttr(indexingMapsAttrs);

    // 迭代器类型：m和n是并行的，k是归约维度
    SmallVector<Attribute> iteratorTypes = {
      vector::IteratorTypeAttr::get(context, vector::IteratorType::parallel),    // m
      vector::IteratorTypeAttr::get(context, vector::IteratorType::reduction),   // k (reduction)
      vector::IteratorTypeAttr::get(context, vector::IteratorType::parallel)     // n
    };
    auto iterTypesAttr = rewriter.getArrayAttr(iteratorTypes);

    auto kindAttr = vector::CombiningKindAttr::get(
        context, vector::CombiningKind::ADD);

    // 创建 vector.contract 操作，注意输出类型应该是 vector 类型
    auto contractOp = rewriter.create<vector::ContractionOp>(
        loc, outputVectorType, lhs, rhs, output,
        indexingMaps, iterTypesAttr, kindAttr);

    // 将结果转换回tensor类型
    auto result = rewriter.create<UnrealizedConversionCastOp>(
        loc, outputTensorType, contractOp.getResult());

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

// 通用算术运算到 Vector 的转换模板
template <typename SourceOp>
class ArithmeticOpConverter : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  ArithmeticOpConverter(const TypeConverter &typeConverter,
                        MLIRContext *context)
      : OpConversionPattern<SourceOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 检查结果是否为 tensor 类型
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType)
      return failure();

    // 获取操作数并转换为vector
    SmallVector<Value> vectorOperands;
    for (auto operand : adaptor.getOperands()) {
      if (isa<RankedTensorType>(operand.getType())) {
        auto vectorType = VectorType::get(
            cast<RankedTensorType>(operand.getType()).getShape(),
            cast<RankedTensorType>(operand.getType()).getElementType());
        auto cast = rewriter.create<UnrealizedConversionCastOp>(
            op.getLoc(), vectorType, operand);
        vectorOperands.push_back(cast->getResult(0));
      } else {
        vectorOperands.push_back(operand);
      }
    }

    // 创建对应的 vector 类型
    auto vectorType =
        VectorType::get(resultType.getShape(), resultType.getElementType());

    // 创建对应的 vector 操作
    auto vectorOp =
        rewriter.create<SourceOp>(op.getLoc(), vectorType, vectorOperands);

    rewriter.replaceOp(op, vectorOp);
    return success();
  }
};

// 为常见算术运算定义具体的转换类型
using AddOpConverter = ArithmeticOpConverter<arith::AddIOp>;
using SubOpConverter = ArithmeticOpConverter<arith::SubIOp>;
using MulOpConverter = ArithmeticOpConverter<arith::MulIOp>;
using DivOpConverter = ArithmeticOpConverter<arith::DivSIOp>;

// 浮点运算的转换
using AddFOpConverter = ArithmeticOpConverter<arith::AddFOp>;
using SubFOpConverter = ArithmeticOpConverter<arith::SubFOp>;
using MulFOpConverter = ArithmeticOpConverter<arith::MulFOp>;
using DivFOpConverter = ArithmeticOpConverter<arith::DivFOp>;

// 空tensor的转换
class EmptyOpConverter : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  EmptyOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tensor::EmptyOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // // 检查是否有 "tl_tensor" 属性
    // if (!op->hasAttr("tl_tensor")) {
    //   return rewriter.notifyMatchFailure(
    //       op, "tensor.empty operation must have 'tl_tensor' attribute");
    // }
    Operation *replaceOp = op;
    auto loc = op.getLoc();
    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elementType = tensorType.getElementType();
    auto shape = tensorType.getShape();

    // 创建 memref 类型 (使用 workgroup 内存空间)
    auto addressSpace = gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    auto memrefType =
        MemRefType::get(shape, elementType,
                        AffineMap::getMultiDimIdentityMap(tensorType.getRank(),
                                                          op->getContext()),
                        addressSpace);

    // 创建 memref.alloc
    auto alloc = rewriter.create<memref::AllocaOp>(loc, memrefType);

    // 创建对应的 vector 类型
    auto vectorType = VectorType::get(shape, elementType);

    // 检查是否有 fill 操作
    if (auto fillOp = dyn_cast<linalg::FillOp>(*(op->getUsers().begin()))) {
      // 获取填充值并创建 broadcast
      auto value = fillOp.getInputs()[0];
      auto broadcastVec =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, value);

      // 创建索引数组用于 store
      SmallVector<Value> indices(
          shape.size(), rewriter.create<arith::ConstantIndexOp>(loc, 0));

      // 将广播后的向量存入 memref
      rewriter.create<vector::StoreOp>(loc, broadcastVec, alloc, indices);

      // 删除empty op
      rewriter.eraseOp(op);
      replaceOp = fillOp;
    }

    // 创建索引数组用于 load
    SmallVector<Value> loadIndices(
        shape.size(), rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // 从 memref 加载向量
    auto loadedVec =
        rewriter.create<vector::LoadOp>(loc, vectorType, alloc, loadIndices);

    // 将向量转换回 tensor
    auto result = rewriter.create<UnrealizedConversionCastOp>(
        loc, tensorType, loadedVec.getResult());

    rewriter.replaceOp(replaceOp, result);
    return success();
  }
};

// 添加新的TypeConverter类
class TensorToVectorTypeConverter : public TypeConverter {
public:
  TensorToVectorTypeConverter() {
    // 添加默认转换
    addConversion([](Type type) { return type; });

    // 添加tensor到vector的转换
    addConversion([](RankedTensorType tensorType) -> Type {
      return VectorType::get(tensorType.getShape(),
                             tensorType.getElementType());
    });

    // 添加materialization
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
    addTargetMaterialization([](OpBuilder &builder, TypeRange resultTypes,
                                ValueRange inputs,
                                Location loc) -> SmallVector<Value> {
      return builder
          .create<UnrealizedConversionCastOp>(loc, resultTypes, inputs.front())
          ->getResults();
    });
  }
};

// 修改ConvertTritonStructuredToVector类
class ConvertTritonStructuredToVector
    : public triton::impl::ConvertTritonStructuredToVectorBase<
          ConvertTritonStructuredToVector> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<vector::VectorDialect, func::FuncDialect, arith::ArithDialect,
                memref::MemRefDialect, math::MathDialect, linalg::LinalgDialect,
                scf::SCFDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // 创建并配置TypeConverter
    TensorToVectorTypeConverter typeConverter;

    // 设置合法的操作
    target.addLegalDialect<vector::VectorDialect, arith::ArithDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           math::MathDialect, linalg::LinalgDialect,
                           gpu::GPUDialect, func::FuncDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    // 设置需要转换的操作
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { return typeConverter.isLegal(op); });

    target.addDynamicallyLegalOp<scf::YieldOp>(
        [&](scf::YieldOp op) { return typeConverter.isLegal(op); });

    target.addDynamicallyLegalOp<arith::TruncIOp, arith::TruncFOp>(
        [](Operation *op) {
          return !mlir::isa<RankedTensorType>(op->getResult(0).getType());
        });

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          return !mlir::isa<RankedTensorType>(op->getResult(0).getType());
        });

    target.addDynamicallyLegalDialect<linalg::LinalgDialect>([](Operation *op) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp || linalgOp->getNumOperands() == 0)
        return true;
      return !mlir::isa<RankedTensorType>(linalgOp->getOperand(0).getType());
    });

    // 添加转换模式
    RewritePatternSet patterns(context);

    // 添加 SCF 相关的转换模式
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    // 添加其他转换模式
    patterns.add<ArithTypeConversionPattern<arith::TruncFOp>,
                 ArithTypeConversionPattern<arith::TruncIOp>,
                 ArithTypeConversionPattern<arith::ExtFOp>,
                 ArithTypeConversionPattern<arith::ExtSIOp>>(typeConverter,
                                                             context);
    patterns.add<TransferReadOpConversion, TransferWriteOpConversion>(
        typeConverter, context);
    patterns.add<MatmulOpConverter>(context);

    // 添加算术运算转换模式
    patterns.add<AddOpConverter, SubOpConverter, MulOpConverter, DivOpConverter,
                 AddFOpConverter, SubFOpConverter, MulFOpConverter,
                 DivFOpConverter>(typeConverter, context);

    // 添加 EmptyOp 转换模式
    patterns.add<EmptyOpConverter>(context);

    // 应用转换
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }

    PassManager pm(&getContext(), getOperation().getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createConvertTritonStructuredToVectorPass() {
  return std::make_unique<ConvertTritonStructuredToVector>();
}