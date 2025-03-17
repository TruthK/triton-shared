#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::tts;

void mlir::tts::TritonStructuredDialect::printAttribute(Attribute attr,
                                                       DialectAsmPrinter &printer) const {
  if (auto tritonPtrAttr = dyn_cast<TritonPtrAttr>(attr)) {
    printer << "triton_ptr";
    return;
  }
}
Attribute mlir::tts::TritonStructuredDialect::parseAttribute(DialectAsmParser &parser,
                                                  Type type) const {
  if (parser.parseKeyword("triton_ptr"))
    return Attribute();
  return TritonPtrAttr::get(parser.getContext());
}

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void TritonStructuredDialect::initialize() {
  initializeTritonStructuredAttrs();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"

#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp.inc"
