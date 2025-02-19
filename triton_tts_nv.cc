#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
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
using namespace mlir;
void init_triton_triton_shared(py::module &&m) {
  ADD_PASS_WRAPPER_0("triton_to_structured",
                     triton::createTritonToStructuredPass);
  ADD_PASS_WRAPPER_0("triton_to_unstructured",
                     triton::createTritonToUnstructuredPass);
  ADD_PASS_WRAPPER_0("triton_arith_to_linalg",
                     triton::createTritonArithToLinalgPass);
  ADD_PASS_WRAPPER_0("structured_to_memref",
                     triton::createStructuredToMemrefPass);
  ADD_PASS_WRAPPER_0("unstructured_to_memref",
                     triton::createUnstructuredToMemrefPass);
  ADD_PASS_WRAPPER_0("triton_ptr_to_memref",
                     triton::createTritonPtrToMemrefPass);
  ADD_PASS_WRAPPER_0("reconcile_unrealized_casts",
                     createReconcileUnrealizedCastsPass);
}

void init_triton_triton_shared_to_llvmir(py::module &&m) {
  ADD_PASS_WRAPPER_0("linalg_to_affine_loops",
                     createConvertLinalgToAffineLoopsPass);
  ADD_PASS_WRAPPER_0("empty_tensor_to_alloc_tensor",
                     bufferization::createEmptyTensorToAllocTensorPass);
  ADD_PASS_WRAPPER_0("one_shot_bufferize",
                     bufferization::createOneShotBufferizePass);
  ADD_PASS_WRAPPER_0("lower_affine", createLowerAffinePass);
  ADD_PASS_WRAPPER_0("linalg_to_loops", createConvertLinalgToLoopsPass);
  ADD_PASS_WRAPPER_0("expand_strided_metadata",
                     memref::createExpandStridedMetadataPass);
  ADD_PASS_WRAPPER_0("convert_scf_to_cf", createConvertSCFToCFPass);

  ADD_PASS_WRAPPER_0("convert_complex_to_llvm", createConvertComplexToLLVMPass);
  ADD_PASS_WRAPPER_0("convert_arith_to_llvm", createArithToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("convert_math_to_llvm", createConvertMathToLLVMPass);
  ADD_PASS_WRAPPER_0("convert_vector_to_llvm", createConvertVectorToLLVMPass);
  ADD_PASS_WRAPPER_0("convert_index_to_llvm", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("convert_func_to_llvm", createConvertFuncToLLVMPass);
  ADD_PASS_WRAPPER_0("memref_expand", memref::createExpandOpsPass);

  ADD_PASS_WRAPPER_0("finalize_mem_ref_to_llvm",
                     createFinalizeMemRefToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("convert_control_flow_to_llvm",
                     createConvertControlFlowToLLVMPass);
}

void init_triton_tts_nv(py::module &&m) {
  m.doc() = "Python bindings to the TTS_NVIDIA Triton backend";
  auto passes = m.def_submodule("passes");
  init_triton_triton_shared(passes.def_submodule("tts"));

  // load dialects
  m.def("load_dialects",
        [](MLIRContext &context) { registerAllDialects(context); });
}
