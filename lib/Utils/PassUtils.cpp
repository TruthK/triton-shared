#include "triton-shared/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pass-utils"


namespace mlir::tts {

void signalFixedPointModified(Operation *rootOp) {
  MLIRContext *context = rootOp->getContext();
  if (!rootOp->hasAttr("iree.fixedpoint.iteration")) {
    LLVM_DEBUG(llvm::dbgs() << "Not signaling fixed-point modification: not "
                               "running under fixed point iterator");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Signalling fixed-point iterator modification");
  rootOp->setAttr("iree.fixedpoint.modified", UnitAttr::get(context));
}

} // namespace mlir::iree_compiler
