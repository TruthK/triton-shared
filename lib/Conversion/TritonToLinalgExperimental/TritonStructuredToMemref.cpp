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
#include "mlir/Dialect/Linalg/Utils/Utils.h"
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
#define GEN_PASS_DEF_CONVERTTRITONSTRUCTUREDTOMEMREF
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// 转换 TTS_TransferReadOp 到 iree_vector_ext.transfer_read
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
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// 转换 TTS_TransferWriteOp 到 IREE::VectorExt::TransferWriteOp
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
    // 获取操作数
    Value dest = adaptor.getBase();
    Value value = adaptor.getValue();
    auto maskDims = op.getMixedMaskDims();

    // 创建默认的indices - 全0
    auto loc = op.getLoc();
    auto baseType = cast<MemRefType>(dest.getType());
    int64_t rank = baseType.getRank();
    baseType.dump();
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
    auto resultType = MemRefType::get(shape, baseType.getElementType(),
                                      AffineMap(), baseType.getMemorySpace());

    // 使用builder创建新的IREE::VectorExt::TransferWriteOp操作
    auto newOp = rewriter.create<mlir::tts::IREE::VectorExt::TransferWriteOp>(
        loc,
        dest,    // 基址
        value,   // 写入值
        indices, // 索引(全0)
        maskDims // mask维度(从原始op获取)
    );

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// 转换 linalg::GenericOp 的 tensor 到 memref
class GenericOpConverter : public OpConversionPattern<linalg::GenericOp> {
public:
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  GenericOpConverter(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<linalg::GenericOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 处理索引映射 - 使用更简单的方式
    // 为每个输入、输出创建一个标识映射（这是一个简化，实际上应该保留原始映射）
    SmallVector<AffineMap> indexingMaps;
    unsigned numInputs = op.getInputs().size();
    unsigned numOutputs = op.getOutputs().size();
    unsigned numLoops = op.getNumLoops();

    // 为每个输入创建标识映射
    for (unsigned i = 0; i < numInputs; ++i) {
      indexingMaps.push_back(
          AffineMap::getMultiDimIdentityMap(numLoops, rewriter.getContext()));
    }

    // 为每个输出创建标识映射
    for (unsigned i = 0; i < numOutputs; ++i) {
      indexingMaps.push_back(
          AffineMap::getMultiDimIdentityMap(numLoops, rewriter.getContext()));
    }

    // 转换迭代器类型 - 使用更简单的方式
    // 根据操作数数量创建适当的迭代器类型
    // 通常 linalg.generic 使用的迭代器是parallel，除非有降维操作
    SmallVector<mlir::utils::IteratorType> iteratorTypes;
    for (unsigned i = 0; i < numLoops; ++i) {
      iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
    }

    // 如果需要精确匹配原始迭代器类型，可以再添加调试代码
    llvm::errs() << "IteratorTypes attribute: " << op.getIteratorTypes()
                 << "\n";

    // 创建GenericOp
    auto newOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/adaptor.getInputs(),
        /*outputs=*/adaptor.getOutputs(),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*regionBuilder=*/nullptr,
        /*attributes=*/ArrayRef<NamedAttribute>{});

    // 手动设置属性
    for (auto attr : op->getAttrs()) {
      // 跳过已经作为参数传递的属性
      if (attr.getName() == "indexing_maps" ||
          attr.getName() == "iterator_types")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin());
    // 替换原始操作
    rewriter.replaceOp(op, newOp.getOutputs());
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


    // 创建 linalg.matmul 操作，输入和输出均为 memref 类型
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{},        // 无返回值
        /*inputs=*/ValueRange{lhs, rhs},          // 输入参数
        /*outputs=*/ValueRange{output},           // 输出参数
        /*attributes=*/ArrayRef<NamedAttribute>{} // 可选属性
    );

    rewriter.replaceOp(op, output);
    return success();
  }
};

// 转换 linalg.fill 操作
class FillOpConverter : public OpConversionPattern<linalg::FillOp> {
public:
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  FillOpConverter(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<linalg::FillOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 获取输入值
    Value value = adaptor.getInputs()[0];
    Value output = adaptor.getOutputs()[0];

    // 创建新的 linalg.fill 操作，使用 memref 类型
    auto fillOp = rewriter.create<linalg::FillOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{},        // 无返回值
        /*inputs=*/ValueRange{value},             // 输入值
        /*outputs=*/ValueRange{output},           // 输出参数
        /*attributes=*/ArrayRef<NamedAttribute>{} // 可选属性
    );

    rewriter.replaceOp(op, output);
    return success();
  }
};

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
        rewriter.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
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
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

// 添加新的TypeConverter类
class TensorTomemrefTypeConverter : public TypeConverter {
public:
  TensorTomemrefTypeConverter(MLIRContext *context) {
    // 添加默认转换
    addConversion([](Type type) { return type; });

    // 添加tensor到vector的转换
    addConversion([context](RankedTensorType tensorType) -> Type {
      auto addressSpace = gpu::AddressSpaceAttr::get(
          context, gpu::GPUDialect::getWorkgroupAddressSpace());
      int64_t offset = 0;
      SmallVector<int64_t> strides;
      int64_t stride = 1;
      auto elementType = tensorType.getElementType();
      auto shape = tensorType.getShape();
      for (int i = shape.size() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        if (shape[i] != ShapedType::kDynamic)
          stride *= shape[i];
      }
      auto stridedLayout = StridedLayoutAttr::get(context, offset, strides);
      return MemRefType::get(tensorType.getShape(),
                             tensorType.getElementType());
    });

    // 定义一个内部函数来处理materialization
    auto materializeCast = [](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
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

// 修改ConvertTritonStructuredToMemref类
class ConvertTritonStructuredToMemref
    : public triton::impl::ConvertTritonStructuredToMemrefBase<
          ConvertTritonStructuredToMemref> {
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
    TensorTomemrefTypeConverter typeConverter(context);

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
      linalgOp.dump();
      return !mlir::isa<RankedTensorType>(linalgOp->getOperand(0).getType());
    });

    // 添加 linalg.generic 的特定约束
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      // 检查所有输入是否都是非 tensor 类型
      for (auto input : op.getInputs()) {
        if (isa<TensorType>(input.getType()))
          return false;
      }
      // 检查所有输出是否都是非 tensor 类型
      for (auto output : op.getOutputs()) {
        if (isa<TensorType>(output.getType()))
          return false;
      }
      return true;
    });

    // 添加 linalg.fill 的合法性检查
    target.addDynamicallyLegalOp<linalg::FillOp>([](linalg::FillOp op) {
      // 检查输出是否为非 tensor 类型
      return !isa<TensorType>(op.getOutputs()[0].getType());
    });

    // 添加转换模式
    RewritePatternSet patterns(context);

    // 添加 SCF 相关的转换模式
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    patterns.add<TransferReadOpConversion, TransferWriteOpConversion>(
        typeConverter, context);
    patterns.add<MatmulOpConverter>(typeConverter, context);
    patterns.add<GenericOpConverter>(typeConverter, context);
    patterns.add<FillOpConverter>(typeConverter, context);

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

    // 消除不必要的 unrealized_conversion_cast
    getOperation()->walk([&](Operation *op) {
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
        // 检查源类型和目标类型是否相同（忽略地址空间）
        auto srcType = castOp.getOperand(0).getType();
        auto dstType = castOp.getResult(0).getType();
        
        if (auto srcMemRef = mlir::dyn_cast<MemRefType>(srcType)) {
          if (auto dstMemRef = mlir::dyn_cast<MemRefType>(dstType)) {
            // 比较除地址空间外的所有属性
            if (srcMemRef.getShape() == dstMemRef.getShape() &&
                srcMemRef.getElementType() == dstMemRef.getElementType()) {
              // 替换所有使用
              castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
              castOp.erase();
            }
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createConvertTritonStructuredToMemrefPass() {
  return std::make_unique<ConvertTritonStructuredToMemref>();
}