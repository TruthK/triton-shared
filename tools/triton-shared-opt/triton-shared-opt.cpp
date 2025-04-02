//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "../RegisterTritonSharedDialects.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonSharedDialects(registry);
registry.insert<mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Triton-Shared test driver\n", registry));
}
