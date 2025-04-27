#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "triton-shared/Codegen/Interfaces/Interfaces.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Codegen/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

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

void init_tts_codegen(py::module &&m) {

  m.def("iree_materialize_target",
        [](mlir::PassManager &pm, int32_t capability, int32_t ptxVersion) {
          mlir::tts::MaterializeTargetPassOptions options;
          options.computeCapability = capability;
          options.ptxVersion = ptxVersion;
          pm.addPass(mlir::tts::createMaterializeTargetPass(options));
        });

  ADD_PASS_WRAPPER_0("iree_llvmgpu_select_lowering_strategy",
                     mlir::tts::createLLVMGPUSelectLoweringStrategyPass);

  m.def("iree_llvmgpu_codegen",
        [](mlir::PassManager &pm, int32_t capability, int32_t ptxVersion) {
          mlir::tts::LLVMGPUCodegenPassOptions options;
          options.computeCapability = capability;
          options.ptxVersion = ptxVersion;
          pm.addPass(mlir::tts::createLLVMGPUCodegenPass(options));
        });
  ADD_PASS_WRAPPER_0("llvmgpu_ptr_transform",
                     mlir::tts::createPtrTransformPass);
}

void init_triton_ttsnv(py::module &&m) {
  m.doc() = "Python bindings to the TTS_NVIDIA Triton backend";
  auto passes = m.def_submodule("passes");
  init_triton_triton_shared(passes.def_submodule("tts"));
  init_tts_codegen(passes.def_submodule("tts_codegen"));
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::ttx::TritonTilingExtDialect,
                    mlir::tts::TritonStructuredDialect,
                    mlir::triton::TritonDialect>();
    mlir::registerAllDialects(registry);
    mlir::tts::registerCodegenPasses();
    mlir::tts::registerCodegenDependentDialects(registry);
    mlir::tts::registerCodegenInterfaces(registry);
    mlir::tts::registerUKernelBufferizationInterface(registry);
    context.appendDialectRegistry(registry);
    mlir::tts::registerTilingInterfaceExternalModels(registry);
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
