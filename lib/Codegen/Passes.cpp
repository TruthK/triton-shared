// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Passes.h"
#include "mlir/Pass/PassManager.h"

//===---------------------------------------------------------------------===//
// Include pass headers per target device
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// #include "triton-shared/Codegen/Common/CPU/Passes.h"
#include "triton-shared/Codegen/Common/GPU/Passes.h"
#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "triton-shared/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Codegen/Transforms/Transforms.h"
#include "triton-shared/Dialect/Encoding/IR/EncodingDialect.h"


#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"


namespace mlir::tts {
void registerTransformDialectTranslationDependentDialects(
    DialectRegistry &registry) {
  // TODO: this is only necessary to make registry subset happy when running
  // the lowering to LLVM. The lowering should be changed to stop using the
  // nested pass manager and this will go away.

  // clang-format off
  registry.insert<mlir::tts::IREE::Encoding::IREEEncodingDialect,
                  mlir::tts::IREE::VectorExt::IREEVectorExtDialect,
                  mlir::tts::IREE::Codegen::IREECodegenDialect,
                  mlir::tts::IREE::GPU::IREEGPUDialect,
                  arith::ArithDialect,
                  affine::AffineDialect,
                  bufferization::BufferizationDialect,
                  func::FuncDialect,
                  gpu::GPUDialect,
                  linalg::LinalgDialect,
                  LLVM::LLVMDialect,
                  scf::SCFDialect,
                  tensor::TensorDialect,
                  transform::TransformDialect,
                  vector::VectorDialect>();
  // clang-format on

  // TODO: these should be registered by the extension instead, but there is
  // no support for it in core currently.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerFindPayloadReplacementOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);

  // registry.addExtensions<
      // mlir::tts::IREE::LinalgExt::LinalgExtTransformOpsExtension,
      // transform_ext::StructuredTransformOpsExtension>();
  // tts::registerTransformDialectCommonExtension(registry);
  // tts::registerTransformDialectFlowExtension(registry);
  // tts::registerTransformDialectLLVMCPUExtension(registry);
  // tts::registerTransformDialectLLVMGPUExtension(registry);
  affine::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  gpu::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerTransformDialectExtension(registry);
  transform::registerLoopExtension(registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);
  vector::registerTransformDialectExtension(registry);
}

void registerCodegenPasses() {
  // Generated.
  registerCodegenCommonPasses();
  // registerCodegenCommonGPUPasses();
  registerCodegenLLVMGPUPasses();
  registerIREEGPUPasses();
  registerIREEVectorExtPasses();
}

void registerCodegenDependentDialects(DialectRegistry &registry)  {

  registry.insert<gpu::GPUDialect, nvgpu::NVGPUDialect,
                  IREE::Codegen::IREECodegenDialect,
                  transform::TransformDialect, IREE::GPU::IREEGPUDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  // Configuration may load and manipulate transform dialect libraries.
  registerTransformDialectTranslationDependentDialects(registry);
}

} // namespace mlir::tts
