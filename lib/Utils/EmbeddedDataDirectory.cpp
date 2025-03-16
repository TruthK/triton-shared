// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Utils/EmbeddedDataDirectory.h"

namespace mlir::tts {

void EmbeddedDataDirectory::withGlobal(
    llvm::function_ref<void(EmbeddedDataDirectory &)> callback) {
  static EmbeddedDataDirectory dir;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  callback(dir);
}

bool EmbeddedDataDirectory::addFile(llvm::StringRef fileName, 
                                    llvm::StringRef contents) {
  auto [_iter, success] = map.insert({fileName, contents});
  return success;
}

std::optional<llvm::StringRef> 
EmbeddedDataDirectory::getFile(llvm::StringRef fileName) const {
  auto iter = map.find(fileName);
  if (iter == map.end()) {
    return std::nullopt;
  }
  return iter->getValue();
}

} // namespace mlir::tts
