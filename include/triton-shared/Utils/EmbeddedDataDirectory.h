// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_
#define IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_

#include <mutex>
#include <optional>
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tts {

class EmbeddedDataDirectory {
public:
  static void withGlobal(llvm::function_ref<void(EmbeddedDataDirectory &)> callback);

  bool addFile(llvm::StringRef fileName, llvm::StringRef contents);
  std::optional<llvm::StringRef> getFile(llvm::StringRef fileName) const;
  llvm::StringMap<llvm::StringRef>& getMap() { return map; }

private:
  llvm::StringMap<llvm::StringRef> map;
};

} // namespace mlir::tts

#endif // IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_
