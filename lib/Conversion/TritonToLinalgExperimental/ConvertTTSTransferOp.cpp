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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#define GEN_PASS_DEF_CONVERTTTSTRANSFEROP
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

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, op.getResult().getType(), newOp.getResult(), true /* restrict */,
        false /* writable */);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

// 转换 TTS_TransferWriteOp 到 IREE::VectorExt::TransferWriteOp
class TransferWriteOpConversion
    : public OpConversionPattern<tts::TransferWriteOp> {
public:
  using OpConversionPattern<tts::TransferWriteOp>::OpConversionPattern;

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

// 修改ConvertTTSTransferOp类
class ConvertTTSTransferOp
    : public triton::impl::ConvertTTSTransferOpBase<ConvertTTSTransferOp> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, memref::MemRefDialect,
                math::MathDialect, linalg::LinalgDialect, scf::SCFDialect,
                ttx::TritonTilingExtDialect, tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // 设置合法的操作
    target.addLegalDialect<
        arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
        math::MathDialect, linalg::LinalgDialect, gpu::GPUDialect,
        func::FuncDialect, IREE::VectorExt::IREEVectorExtDialect>();
    target.addLegalOp<UnrealizedConversionCastOp,
                      IREE::VectorExt::TransferWriteOp,
                      IREE::VectorExt::TransferReadOp>();

    target.addIllegalOp<tts::TransferReadOp, tts::TransferWriteOp>();
    // 添加转换模式
    RewritePatternSet patterns(context);
    patterns.add<TransferReadOpConversion, TransferWriteOpConversion>(
        context); 

    // 应用转换
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
triton::createConvertTTSTransferOpPass() {
  return std::make_unique<ConvertTTSTransferOp>();
}