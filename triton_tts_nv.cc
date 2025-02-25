#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/LinalgToLLVM/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "llvm/IR/Constants.h"

#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

void init_triton_triton_shared(py::module &&m) {
  ADD_PASS_WRAPPER_0("triton_to_linalg",
                     mlir::triton::createTritonToLinalgExperimentalPass);
}

void init_triton_triton_shared_to_llvmir(py::module &&m) {
  m.def("memref_copy_to_linalg", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tts::createMemrefCopyToLinalgPass());
  });
  ADD_PASS_WRAPPER_0("linalg_to_llvm", mlir::tts::createLinalgToLLVMPass);
}

void init_triton_tts_nv(py::module &&m) {
  m.doc() = "Python bindings to the TTS_NVIDIA Triton backend";
  auto passes = m.def_submodule("passes");
  init_triton_triton_shared(passes.def_submodule("tts"));
  init_triton_triton_shared_to_llvmir(passes.def_submodule("convert"));
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::ttx::TritonTilingExtDialect,
                    mlir::tts::TritonStructuredDialect,
                    mlir::triton::TritonDialect>();
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::registerAllPasses();
    mlir::registerAllTranslations();
    context.appendDialectRegistry(registry);
  });

  // TODO: could be done in python if we had a generic interface to set metadata
  m.def("set_nvvm_reflect_ftz", [](llvm::Module *mod) {
    // please check https://llvm.org/docs/NVPTXUsage.html#reflection-parameters
    // this will enable fast math path in libdevice
    // for example, when enable nvvm-reflect-ftz, sqrt.approx.f32 will change to
    // sqrt.approx.ftz.f32
    using namespace llvm;
    auto &ctx = mod->getContext();
    Type *i32 = Type::getInt32Ty(ctx);
    auto *mdFour = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 4));
    auto *mdName = MDString::get(ctx, "nvvm-reflect-ftz");
    auto *mdOne = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 1));
    auto *reflect = MDNode::get(ctx, {mdFour, mdName, mdOne});
    mod->addModuleFlag(reflect);
  });
}
