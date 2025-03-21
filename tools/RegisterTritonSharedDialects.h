#pragma once
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"

#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton-shared/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "triton-shared/Codegen/Interfaces/Interfaces.h"
#include "triton-shared/Codegen/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace test
} // namespace mlir

inline void registerTritonSharedDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);

  registry.insert<mlir::ttx::TritonTilingExtDialect,
                  mlir::tts::TritonStructuredDialect,
                  mlir::triton::TritonDialect>();
  mlir::registerAllExtensions(registry);

  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::registerLinalgPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToLinalgExperimentalPass();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonPtrToMemref();
  mlir::triton::registerUnstructuredToMemref();
  mlir::triton::registerTritonToUnstructuredPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerStructuredToMemrefPasses();
  
  mlir::tts::registerCodegenDependentDialects(registry);
  mlir::tts::registerCodegenInterfaces(registry);
  mlir::tts::registerUKernelBufferizationInterface(registry);
  mlir::tts::registerCodegenPasses();
}
