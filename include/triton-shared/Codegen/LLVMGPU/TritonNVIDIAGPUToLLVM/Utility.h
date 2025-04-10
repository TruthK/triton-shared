#ifndef TRITON_TTS_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_UTILITY_H
#define TRITON_TTS_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_UTILITY_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define DEBUG_TYPE "tts_ttgpu_to_llvm"

using namespace mlir;

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators

namespace mlir {
namespace LLVM {
namespace tts {
namespace NVIDIA {

Value getSRegValue(OpBuilder &b, Location loc, StringRef sRegStr);
// Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
// Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i);
// Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i);
// Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i);
// Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
//   Value mask);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis);

inline bool isKernel(FunctionOpInterface funcOp) {
  return funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

/// Create a predicate with just single active thread.
// Value createElectPredicate(Location loc, RewriterBase &rewriter);
// Value createElectPredicateWarp0(Location loc, RewriterBase &rewriter);

// Create bar.warp.sync
// void createSyncWarp(Location loc, OpBuilder &builder);

} // namespace NVIDIA
} // namespace tts
} // namespace LLVM
} // namespace mlir

#endif
