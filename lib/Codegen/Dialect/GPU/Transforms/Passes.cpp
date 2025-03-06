

#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::tts {

namespace tts::GPU {
namespace {
#define GEN_PASS_REGISTRATION
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace
} // namespace tts::GPU

void registerIREEGPUPasses() {
  // Generated.
  tts::GPU::registerPasses();
}
} // namespace mlir::tts
