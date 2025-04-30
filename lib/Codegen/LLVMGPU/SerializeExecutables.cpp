#include <memory>
#include <utility>

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Codegen/Utils/GPUUtils.h"
#include "triton-shared/Codegen/Utils/Utils.h"

#include "lib/Target/LLVMIR/LLVMPasses.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"

#define DEBUG_TYPE "iree-serialize"

namespace mlir::tts {

#define GEN_PASS_DEF_SERIALIZETARGETEXECUTABLESPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct CUDAOptions {
  std::string clTarget = "sm_89";
  std::string clTargetFeatures = "+ptx84";
};

std::string sanitizeSymbolName(StringRef name) {
  std::string result;
  result.reserve(name.size());
  for (size_t i = 0; i < name.size(); ++i) {
    char c = name[i];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_')) {
      c = '_';
    }
    result.push_back(c);
  }
  return result;
}

static std::string translateModuleToISA(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine) {

  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());
  pm.run(module);

  std::string targetISA;
  {
    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CodeGenFileType::AssemblyFile);
    codegenPasses.run(module);
  }
  return targetISA;
}

/// Performs optimizations on |module| (including LTO-style whole-program ones).
static void optimizeModule(llvm::Module *mod,
                           llvm::TargetMachine &targetMachine,
                           const std::array<int32_t, 3> &maxWorkgroupSize) {

  using namespace llvm;
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  PassInstrumentationCallbacks *instrCbPtr = nullptr;
  PassInstrumentationCallbacks passInstrCb;
  StandardInstrumentations standardInstr(mod->getContext(),
                                         /*DebugLogging*/ true);

  PipelineTuningOptions tuningOptions;
  tuningOptions.LoopUnrolling = true;
  tuningOptions.LoopInterleaving = true;
  tuningOptions.LoopVectorization = true;
  // TODO: currently we run SLP vectorizer with an empty target machine.
  // This cause the vectorizer to create larger vector which could be bad.
  // Disabling it would currently cause regressions as this pass also
  // applies some scheduling that helps performance in some cases. We
  // should work on using NVPTX target instead and address the performance
  // regressions with some scheduling solution.
  tuningOptions.SLPVectorization = true;

  PassBuilder pb(/*targetMachine=*/&targetMachine, tuningOptions, std::nullopt,
                 instrCbPtr);

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  ModulePassManager mpm;
  pb.registerVectorizerStartEPCallback(
      [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
        // Triton generates large structure of scalars which may pessimise
        // optimizations, we run a pass to break up phi of struct to make
        // sure all the struct are removed for the following passes.
        fpm.addPass(BreakStructPhiNodesPass());
        fpm.addPass(InstCombinePass());
      });
  mpm.addPass(pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3));
  mpm.run(*mod, mam);
}

LogicalResult serializeExecutable(std::string &targetPTX,
                                  const CUDAOptions &options,
                                  ModuleOp innerModuleOp,
                                  OpBuilder &executableBuilder) {
  StringRef targetArch = options.clTarget;
  StringRef targetFeatures = options.clTargetFeatures;
  if (auto attr = getGPUTargetAttr(innerModuleOp)) {
    targetArch = attr.getArch();
    targetFeatures = attr.getFeatures();
  }

  // We name our files after the executable name so that they are easy to
  // track both during compilation (logs/artifacts/etc), as outputs (final
  // intermediate code/binary files), and at runtime (loaded
  // libraries/symbols/etc).
  // auto libraryName =
  //     innerModuleOp->getParentOfType<LLVM::LLVMFuncOp>().getName().str();

  std::array<int32_t, 3> maxWorkgroupSize = {1, 1, 1};
  // Perform the translation in a separate context to avoid any
  // multi-threading issues.
  llvm::LLVMContext context;

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(innerModuleOp, context);
  if (!llvmModule) {
    return innerModuleOp.emitError() << "failed to translate the MLIR LLVM "
                                        "dialect to the native llvm::Module";
  }

  for (auto funcOp : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
    llvm::Function *llvmFunc = llvmModule->getFunction(funcOp.getName());
    if (llvmFunc->isDeclaration()) {
      continue;
    }

    // Sanitize the function name as PTX has strict requirements.
    llvmFunc->setName(sanitizeSymbolName(funcOp.getName()));

    auto *annotations =
        llvmModule->getOrInsertNamedMetadata("nvvm.annotations");
    auto setMetadataValueI32 = [&](StringRef name, int value) {
      llvm::Metadata *llvmMetadata[] = {
          llvm::ValueAsMetadata::get(llvmFunc),
          llvm::MDString::get(llvmModule->getContext(), name),
          llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(llvmModule->getContext()), value))};
      annotations->addOperand(
          llvm::MDNode::get(llvmModule->getContext(), llvmMetadata));
    };

    // Mark the entry point as a kernel.
    setMetadataValueI32("kernel", 1);

  }

  std::unique_ptr<llvm::TargetMachine> targetMachine;
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXAsmPrinter();
  {
    llvm::Triple triple("nvptx64-nvidia-cuda");
    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      return innerModuleOp.emitError() << "cannot initialize target triple";
    }
    targetMachine.reset(target->createTargetMachine(triple.str(), targetArch,
                                                    targetFeatures, {}, {}));
    if (targetMachine == nullptr) {
      return innerModuleOp.emitError() << "cannot initialize target machine";
    }
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  // Run LLVM optimization passes.
  optimizeModule(llvmModule.get(), *targetMachine, maxWorkgroupSize);
  // Serialize ptx kernel into the binary that we will embed in the
  // final FlatBuffer.
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::store:\n";
    llvmModule->dump();
  });

  targetPTX = translateModuleToISA(*llvmModule, *targetMachine);
  if (targetPTX.empty()) {
    return failure();
  }
  llvm::outs() << "targetPTX: "
               << "\n"
               << targetPTX;
  llvm::outs() << "\n"
               << "end targetPTX: "
               << "\n";

  return success();
}

struct SerializeTargetExecutablesPass
    : public impl::SerializeTargetExecutablesPassBase<
          SerializeTargetExecutablesPass> {
  using impl::SerializeTargetExecutablesPassBase<
      SerializeTargetExecutablesPass>::SerializeTargetExecutablesPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NVVM::NVVMDialect, LLVM::LLVMDialect,
                    IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    OpBuilder executableBuilder(moduleOp);
    std::string targetPTX;
    // Ask the target backend to serialize the executable. Note that it
    // may create one or more hal.executable.binary ops in the case of
    // multi-architecture binaries.
    if (failed(serializeExecutable(targetPTX, options, moduleOp,
                                   executableBuilder))) {
      moduleOp.emitError()
          << "failed to serialize executable for target backend ";
      return signalPassFailure();
    }
    // moduleOp->setAttr("tt.ptx_code", StringAttr::get(moduleOp.getContext(), targetPTX));
  }

private:
  CUDAOptions options;
};

} // namespace

} // namespace mlir::tts
