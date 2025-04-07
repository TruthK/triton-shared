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
#include "mlir/IR/AffineMap.h"
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

#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
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

// 将函数声明移到类外部
LogicalResult convertUnrealizedCastToVectorLoad(Operation *op,
                                                OpBuilder &builder) {
  auto castOp = dyn_cast<UnrealizedConversionCastOp>(op);
  if (!castOp)
    return failure();

  // 检查是否是从 memref 到 vector 的转换
  if (castOp.getNumOperands() != 1)
    return failure();

  auto memrefType = mlir::dyn_cast<MemRefType>(castOp.getOperand(0).getType());
  auto vectorType = mlir::dyn_cast<VectorType>(castOp.getResult(0).getType());

  if (!memrefType || !vectorType)
    return failure();

  // 创建全0索引数组
  auto loc = castOp.getLoc();
  SmallVector<Value> indices(memrefType.getRank(),
                             builder.create<arith::ConstantIndexOp>(loc, 0));

  // 创建 vector.load 操作
  auto loadOp = builder.create<vector::LoadOp>(loc, vectorType,
                                               castOp.getOperand(0), indices);

  // 替换操作
  castOp->replaceAllUsesWith(loadOp);
  castOp->erase();

  return success();
}

// 转换 TTS_TransferReadOp 到 vector.transfer_read
class TransferReadOpConversion
    : public OpConversionPattern<tts::TransferReadOp> {
public:
  using OpConversionPattern<tts::TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::TransferReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 获取操作数
    Value base = adaptor.getBase();
    auto maskDims = op.getMixedMaskDims();
    Value other = adaptor.getOther();

    // 创建默认的indices - 全0
    auto loc = op.getLoc();
    auto baseType = cast<MemRefType>(base.getType());
    int64_t rank = baseType.getRank();

    // 创建OpFoldResult数组,全部使用静态0
    SmallVector<OpFoldResult> indices;
    indices.resize(rank, rewriter.getI64IntegerAttr(0));

    // 创建带有 stride 的结果类型
    auto shape = baseType.getShape();
    int64_t offset = 0;
    SmallVector<int64_t> strides;
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), stride);
      if (shape[i] != ShapedType::kDynamic)
        stride *= shape[i];
    }
    auto stridedLayout =
        StridedLayoutAttr::get(rewriter.getContext(), offset, strides);

    // 创建新的 memref 类型，保持原有的地址空间
    // auto resultType = MemRefType::get(shape, baseType.getElementType(),
    //                                   stridedLayout,
    //                                   baseType.getMemorySpace());
    auto resultType = MemRefType::get(shape, baseType.getElementType(),
                                      AffineMap(), baseType.getMemorySpace());

    // 使用builder创建新的操作
    auto newOp = rewriter.create<mlir::tts::IREE::VectorExt::TransferReadOp>(
        loc,
        resultType, // 带有 stride 的结果类型
        base,       // 基址
        indices,    // 索引
        maskDims,   // mask维度
        other       // 其他值
    );
    newOp->dump();
    rewriter.replaceOp(op, newOp.getResult());
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

class MatmulOpConverter : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  MatmulOpConverter(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<linalg::MatmulOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 获取输入和输出值
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];

    // // 对 lhs 做 vector -> memref 转换
    // if (!isa<MemRefType>(lhs.getType())) {
    //   auto tensorType = cast<RankedTensorType>(lhs.getType());
    //   auto memrefType =
    //       MemRefType::get(tensorType.getShape(),
    //       tensorType.getElementType());
    //   lhs = rewriter.create<UnrealizedConversionCastOp>(loc, memrefType, lhs)
    //             .getResult(0);
    // }

    // // 对 rhs 做 vector -> memref 转换
    // if (!isa<MemRefType>(rhs.getType())) {
    //   auto tensorType = cast<RankedTensorType>(rhs.getType());
    //   auto memrefType =
    //       MemRefType::get(tensorType.getShape(),
    //       tensorType.getElementType());
    //   rhs = rewriter.create<UnrealizedConversionCastOp>(loc, memrefType, rhs)
    //             .getResult(0);
    // }

    if (!isa<MemRefType>(output.getType())) {
      while (output.getDefiningOp()->hasAttr("dot.result_tensor_type")) {
        if (output.getDefiningOp<memref::AllocOp>())
          break;
        output = output.getDefiningOp()->getOperand(0);
      }
    }

    // 创建 linalg.matmul 操作，输入和输出均为 memref 类型
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{},        // 无返回值
        /*inputs=*/ValueRange{lhs, rhs},          // 输入参数
        /*outputs=*/ValueRange{output},           // 输出参数
        /*attributes=*/ArrayRef<NamedAttribute>{} // 可选属性
    );
    matmulOp->dump();
    // 获取结果 memref 类型
    auto resultMemrefType = cast<MemRefType>(output.getType());

    // 创建对应的 vector 类型
    auto resultVectorType = VectorType::get(resultMemrefType.getShape(),
                                            resultMemrefType.getElementType());

    // 将结果从 memref 转换为 vector 类型
    auto result = rewriter.create<UnrealizedConversionCastOp>(
        loc, resultVectorType, output);

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
    Operation *replaceOp = op;
    auto loc = op.getLoc();
    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elementType = tensorType.getElementType();
    auto shape = tensorType.getShape();

    // 创建 strided layout
    int64_t offset = 0;
    SmallVector<int64_t> strides;
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), stride);
      if (shape[i] != ShapedType::kDynamic)
        stride *= shape[i];
    }
    auto stridedLayout =
        StridedLayoutAttr::get(rewriter.getContext(), offset, strides);

    // 创建 memref 类型 (使用 workgroup 内存空间和 strided layout)
    auto addressSpace = gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    // auto memrefType =
    //     MemRefType::get(shape, elementType, stridedLayout, addressSpace);
    auto memrefType =
        MemRefType::get(shape, elementType, AffineMap(), addressSpace);

    // 复制原始 EmptyOp 的所有属性
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs()) {
      attrs.push_back(attr);
    }

    // 创建 memref.alloc 并保留属性
    auto alloc = rewriter.create<memref::AllocOp>(loc, memrefType,
                                                  ValueRange{}, // 无动态尺寸
                                                  attrs); // 保留原属性

    // 创建对应的 vector 类型
    auto vectorType = VectorType::get(shape, elementType);

    // 检查是否有 fill 操作
    if (auto fillOp = dyn_cast<linalg::FillOp>(*(op->getUsers().begin()))) {
      // 获取填充值并创建 broadcast
      auto value = fillOp.getInputs()[0];
      // 保留 fillOp 的属性
      SmallVector<NamedAttribute> fillAttrs;
      for (auto attr : fillOp->getAttrs()) {
        fillAttrs.push_back(attr);
      }
      auto broadcastVec = rewriter.create<vector::BroadcastOp>(
          loc, vectorType, value, fillAttrs);

      // 创建索引数组用于 store
      SmallVector<Value> indices(
          shape.size(), rewriter.create<arith::ConstantIndexOp>(loc, 0));

      // 将广播后的向量存入 memref
      auto storeOp = rewriter.create<vector::StoreOp>(
          loc, broadcastVec, alloc, indices, /*nontemporal=*/false);
      // 如果需要，可以在创建后设置属性
      for (auto attr : fillAttrs) {
        storeOp->setAttr(attr.getName(), attr.getValue());
      }

      // 删除empty op
      rewriter.eraseOp(op);
      replaceOp = fillOp;
    }

    // 创建索引数组用于 load
    SmallVector<Value> loadIndices(
        shape.size(), rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // 从 memref 加载向量
    auto loadOp = rewriter.create<vector::LoadOp>(
        loc, vectorType, alloc, loadIndices, /*nontemporal=*/false);
    // 如果需要，可以在创建后设置属性
    for (auto attr : attrs) {
      loadOp->setAttr(attr.getName(), attr.getValue());
    }

    // 将向量转换回 tensor，保留属性
    auto result = rewriter.create<UnrealizedConversionCastOp>(
        loc, tensorType, loadOp.getResult(), attrs);

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

    // 定义一个内部函数来处理materialization
    auto materializeCast = [](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
      // 检查是否是从 memref 到 vector 的转换
      if (auto memrefType = mlir::dyn_cast<MemRefType>(inputs[0].getType())) {
        if (auto vectorType = mlir::dyn_cast<VectorType>(resultType)) {
          // 创建全0索引数组
          SmallVector<Value> indices(
              memrefType.getRank(),
              builder.create<arith::ConstantIndexOp>(loc, 0));

          // 使用 vector.load
          return builder.create<vector::LoadOp>(loc, vectorType, inputs[0],
                                                indices);
        }
      }

      // 其他情况使用默认的 UnrealizedConversionCastOp
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    // 添加source materialization
    addSourceMaterialization(materializeCast);
    // 添加target materialization
    addTargetMaterialization(materializeCast);
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
                           gpu::GPUDialect, func::FuncDialect,
                           IREE::VectorExt::IREEVectorExtDialect>();
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
      llvm::errs() << "linalgOp->getOperand(0).getType(): "
                   << linalgOp->getOperand(0).getType() << "\n";
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
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
    getOperation()->dump();

    // 遍历所有操作并应用转换
    getOperation()->walk([&](Operation *op) {
      OpBuilder builder(op);
      if (succeeded(convertUnrealizedCastToVectorLoad(op, builder))) {
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createConvertTritonStructuredToVectorPass() {
  return std::make_unique<ConvertTritonStructuredToVector>();
}