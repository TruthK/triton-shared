#ifndef TRITON_CONVERSION_STRUCTUREDTOMEMREF_STRUCTUREDTOMEMREF_H
#define TRITON_CONVERSION_STRUCTUREDTOMEMREF_STRUCTUREDTOMEMREF_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"

void populateStructuredToMemrefConversionPatterns(RewritePatternSet &patterns,
                                                  TypeConverter &typeConverter);

void populateSimplifyTTSMakeTPtrPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createStructuredToMemrefPass();
std::unique_ptr<OperationPass<ModuleOp>> createSimplifyTTSMakeTPtrPass();
std::unique_ptr<OperationPass<func::FuncOp>> createHoistTensorEmptyFromLoopsPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_STRUCTUREDTOMEMREF_STRUCTUREDTOMEMREF_H
