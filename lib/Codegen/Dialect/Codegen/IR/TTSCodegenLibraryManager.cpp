
#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Parser/Parser.h"

namespace mlir::tts::GPU {

/// Helper function that implements the module lookup and validation.
/// The caller is responsible for acquiring the mutex for `libraryModules".
static FailureOr<ModuleOp> getOrParseTransformLibraryModuleImpl(
    StringRef libraryPath,
    llvm::StringMap<OwningOpRef<ModuleOp>> &libraryModules,
    llvm::function_ref<LogicalResult(OwningOpRef<ModuleOp> &)> loadModuleFn) {

  auto loadedLibrary = libraryModules.find(libraryPath);
  if (loadedLibrary != libraryModules.end()) {
    if (ModuleOp module = loadedLibrary->second.get()) {
      return module;
    }
    return failure();
  }

  OwningOpRef<ModuleOp> &parsedLibrary = libraryModules[libraryPath];

  if (failed(loadModuleFn(parsedLibrary))) {
    return failure();
  }

  if (!parsedLibrary.get()->hasAttr(
          transform::TransformDialect::kWithNamedSequenceAttrName)) {
    parsedLibrary->emitError()
        << "Module without the '"
        << transform::TransformDialect::kWithNamedSequenceAttrName
        << "' attribute is not a transform dialect library";

    parsedLibrary = nullptr;
    return failure();
  }

  if (!parsedLibrary->getSymName()) {
    parsedLibrary->setSymName("__transform");
  }

  return parsedLibrary.get();
}

FailureOr<ModuleOp>
IREECodegenDialect::getOrLoadTransformLibraryModule(StringRef libraryPath) {
  std::lock_guard<std::mutex> guard(libraryMutex);
  MLIRContext *ctx = getContext();

  return getOrParseTransformLibraryModuleImpl(
      libraryPath, libraryModules, [=](OwningOpRef<ModuleOp> &parsedLibrary) {
        return transform::detail::parseTransformModuleFromFile(ctx, libraryPath,
                                                               parsedLibrary);
      });
}

FailureOr<ModuleOp> IREECodegenDialect::getOrParseTransformLibraryModule(
    StringRef libraryPath, StringRef libraryMLIRSource) {
  std::lock_guard<std::mutex> guard(libraryMutex);
  MLIRContext *ctx = getContext();

  return getOrParseTransformLibraryModuleImpl(
      libraryPath, libraryModules, [=](OwningOpRef<ModuleOp> &parsedLibrary) {
        ParserConfig config(ctx);
        parsedLibrary =
            parseSourceString<ModuleOp>(libraryMLIRSource, ctx, libraryPath);
        return success(*parsedLibrary != nullptr);
      });
}

} // namespace mlir::tts::GPU
