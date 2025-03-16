//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "../RegisterTritonSharedDialects.h"
#include "triton-shared/Codegen/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonSharedDialects(registry);
  mlir::tts::registerCodegenPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Triton-Shared test driver\n", registry));
}
