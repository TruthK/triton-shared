
#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenDialect.h"

#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenAttrs.h"
#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenDialect.cpp.inc"
#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenOps.h"
#include "triton-shared/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::tts::GPU {

struct IREECodegenDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<TranslationInfoAttr>(attr)) {
      os << "translation";
      return AliasResult::OverridableAlias;
    } else if (llvm::isa<CompilationInfoAttr>(attr)) {
      os << "compilation";
      return AliasResult::OverridableAlias;
    } else if (llvm::isa<LoweringConfigAttr>(attr)) {
      os << "config";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREECodegenDialect::initialize() {
  initializeCodegenAttrs();
  addInterfaces<IREECodegenDialectOpAsmInterface>();

  addOperations<
#define GET_OP_LIST
#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Codegen/Dialect/Codegen/IR/UKernelOps.cpp.inc"
      >();
}

LogicalResult
IREECodegenDialect::verifyOperationAttribute(Operation *op,
                                             NamedAttribute attribute) {
  StringRef symbol = attribute.getName().strref();
  Attribute attr = attribute.getValue();
  
  if (symbol == kTuningSpecDefaultEntrypointAttrName) {
    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
      if (!llvm::any_of(moduleOp.getOps<transform::NamedSequenceOp>(),
                        [](transform::NamedSequenceOp op) {
                          return op.getName() == kKernelConfigSpecName;
                        })) {
        return moduleOp.emitError()
               << "The tuning specification must include a named "
                  "sequence with the symbol name '"
               << kKernelConfigSpecName << "'.";
      }
    }
  }

  if (symbol != kTuningSpecEntrypointAttrName)
    return success();

  if (!isa<UnitAttr>(attr)) {
    return op->emitError("'") << symbol << "' attribute must be a UnitAttr";
  }

  if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(op)) {
    ArrayRef<Type> resTypes = namedSeqOp.getFunctionType().getResults();
    if (resTypes.size() != 1 || !isa<transform::AnyOpType>(resTypes[0])) {
      return namedSeqOp.emitError()
             << "Tuning spec entry point expected to return any_op";
    }

    ArrayRef<Type> argTypes = namedSeqOp.getArgumentTypes();
    if (argTypes.size() != 1 || !isa<transform::AnyOpType>(argTypes[0])) {
      return namedSeqOp.emitError()
             << "Tuning spec entry point expected to have a "
                "single any_op argument";
    }
  }

  return success();
}

} // namespace mlir::tts::GPU
