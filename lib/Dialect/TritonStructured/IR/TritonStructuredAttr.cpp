#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::tts;

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredAttr.cpp.inc"


void TritonStructuredDialect::initializeTritonStructuredAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredAttr.cpp.inc" 
      >();
}


