#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "passes.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_triton_shared(py::module &&m) {
  ADD_PASS_WRAPPER_0("triton_to_structured",
                     mlir::triton::createTritonToStructuredPass);
  ADD_PASS_WRAPPER_0("triton_to_unstructured",
                     mlir::triton::createTritonToUnstructuredPass);
  ADD_PASS_WRAPPER_0("triton_arith_to_linalg",
                     mlir::triton::createTritonArithToLinalgPass);
  ADD_PASS_WRAPPER_0("structured_to_memref",
                     mlir::triton::createStructuredToMemrefPass);
  ADD_PASS_WRAPPER_0("unstructured_to_memref",
                     mlir::triton::createUnstructuredToMemrefPass);
  ADD_PASS_WRAPPER_0("triton_ptr_to_memref",
                     mlir::triton::createTritonPtrToMemrefPass);
  ADD_PASS_WRAPPER_0("reconcile_unrealized_casts",
                     mlir::createReconcileUnrealizedCastsPass);
}

void init_triton_tts_nv(py::module &&m) {
  m.doc() = "Python bindings to the TTS_NVIDIA Triton backend";
  auto passes = m.def_submodule("passes");
  init_triton_triton_shared(passes.def_submodule("tts"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry
        .insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::math::MathDialect,
                mlir::linalg::LinalgDialect, mlir::affine::AffineDialect, mlir::scf::SCFDialect,
                mlir::tensor::TensorDialect, mlir::bufferization::BufferizationDialect,
                mlir::memref::MemRefDialect, mlir::ttx::TritonTilingExtDialect,
                mlir::tts::TritonStructuredDialect>();
    mlir::registerNVVMDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
