#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM//PTXAsmFormat.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::tts;

namespace {
// declare vprintf(i8*, i8*) as external function
// LLVM::LLVMFuncOp getVprintfDeclaration(RewriterBase &rewriter) {
//   auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
//   StringRef funcName("vprintf");
//   Operation *funcOp = moduleOp.lookupSymbol(funcName);
//   if (funcOp)
//     return cast<LLVM::LLVMFuncOp>(*funcOp);

//   auto *context = rewriter.getContext();

//   SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context)};
//   auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

//   RewriterBase::InsertionGuard guard(rewriter);
//   rewriter.setInsertionPointToStart(moduleOp.getBody());

//   return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
//                                            funcType);
// }

// // extend integer to int32, extend float to float64
// // this comes from vprintf alignment requirements.
// std::pair<Type, Value> printfPromoteValue(RewriterBase &rewriter, Value value) {
//   auto *context = rewriter.getContext();
//   auto type = value.getType();
//   Value newOp = value;
//   Type newType = type;
//   auto loc = UnknownLoc::get(context);
//   auto b = TritonLLVMOpBuilder(loc, rewriter);

//   bool isUnsigned = type.isUnsignedInteger();
//   if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
//     if (isUnsigned) {
//       newType = ui32_ty;
//       newOp = b.zext(newType, value);
//     } else {
//       newType = i32_ty;
//       newOp = b.sext(newType, value);
//     }
//   } else if (type.isBF16() || type.isF16() || type.isF32()) {
//     newType = f64_ty;
//     newOp = b.fpext(newType, value);
//   }

//   return {newType, newOp};
// }

// LLVM::LLVMFuncOp getAssertfailDeclaration(RewriterBase &rewriter) {
//   auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
//   StringRef funcName("__assertfail");
//   {
//     Operation *funcOp = moduleOp.lookupSymbol(funcName);
//     if (funcOp)
//       return cast<LLVM::LLVMFuncOp>(*funcOp);
//   }
//   // void __assert_fail(const char * assertion, const char * file, unsigned
//   // int line, const char * function);
//   auto *ctx = rewriter.getContext();
//   SmallVector<Type> argsType{ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx),
//                              rewriter.getIntegerType(sizeof(size_t) * 8)};
//   auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);
//   RewriterBase::InsertionGuard guard(rewriter);
//   rewriter.setInsertionPointToStart(moduleOp.getBody());
//   auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx),
//                                                   funcName, funcType);

//   funcOp.setPassthroughAttr(
//       ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));
//   return funcOp;
// }
} // namespace

namespace mlir::tts::NVIDIA {


bool TargetInfo::supportMaximumMinimum() const {
  return computeCapability >= 80;
}


// Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
//                          Value cmp) const {
//   auto b = TritonLLVMOpBuilder(loc, rewriter);
//   Value threadMask = b.int_val(type.getIntOrFloatBitWidth(), -1);
//   return rewriter.create<NVVM::VoteBallotOp>(loc, type, threadMask, cmp);
// }

// static Value mapa(RewriterBase &rewriter, Location loc, Value ptr, Value ctaid,
//                   Value pred) {
//   Value args[] = {ptr, ctaid};
//   StringRef name = "llvm.nvvm.mapa.shared.cluster";
//   return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, ptr.getType(),
//                                          args)
//       .getResult(0);
// }

static std::string getConstraintForBitwidth(unsigned bitwidth) {
  switch (bitwidth) {
  case 8:
  case 16:
    return "h";
  case 32:
    return "r";
  case 64:
    return "l";
  default:
    llvm_unreachable("unsupported bitwidth");
  }
}

// static bool isConstantTruePred(Value pred) {
//   if (auto constOp = pred.getDefiningOp<LLVM::ConstantOp>()) {
//     return cast<IntegerAttr>(constOp.getValue()).getInt() != 0;
//   }
//   return false;
// }

// void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
//                               std::optional<Value> ctaId, Value val,
//                               Value pred) const {
//   auto b = TritonLLVMOpBuilder(loc, rewriter);
//   MLIRContext *ctx = rewriter.getContext();
//   auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
//   assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

//   if (!isa<VectorType>(val.getType())) {
//     storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, {val}, rewriter),
//                  pred);
//     return;
//   }

//   auto vecTy = cast<VectorType>(val.getType());
//   Type elemTy = vecTy.getElementType();
//   unsigned vec = vecTy.getNumElements();
//   unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
//   assert(llvm::isPowerOf2_32(vec));

//   if (elemBitwidth < 8) {
//     assert(vec == 1 &&
//            "don't know how to load/store vectors of sub-byte elems");
//     SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
//     for (Value &v : vals) {
//       v = b.zext(int_ty(8), b.bitcast(v, int_ty(elemBitwidth)));
//     }
//     storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
//                  pred);
//     return;
//   }

//   if (!elemTy.isInteger()) {
//     SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
//     for (Value &v : vals) {
//       v = b.bitcast(v, int_ty(elemBitwidth));
//     }
//     storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
//                  pred);
//     return;
//   }

//   // load/store ops only support v2 and v4.  If the vector width is larger than
//   // 4, we have two strategies for dealing with it.
//   //  1. If the element type is smaller than b32, store b32's instead.
//   //  2. Otherwise, split the store into multiple stores.
//   if (vec > 4 && elemBitwidth < 32) {
//     assert(llvm::isPowerOf2_32(vec));
//     int elemsPerPack = 32 / elemBitwidth;
//     SmallVector<Value> oldVals = unpackLLVector(loc, val, rewriter);

//     SmallVector<Value> newVals;
//     for (int i = 0; i < vec / elemsPerPack; i++) {
//       Value v = packLLVector(
//           loc, ArrayRef(oldVals).slice(i * elemsPerPack, elemsPerPack),
//           rewriter);
//       newVals.push_back(b.bitcast(v, i32_ty));
//     }
//     storeDShared(rewriter, loc, ptr, ctaId,
//                  packLLVector(loc, newVals, rewriter), pred);
//     return;
//   }

//   if (vec * elemBitwidth > 128) {
//     assert(llvm::isPowerOf2_32(vec));
//     assert(elemBitwidth == 32 || elemBitwidth == 64);
//     int maxVec = 128 / elemBitwidth;

//     auto newVecTy = vec_ty(elemTy, maxVec);
//     SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
//     for (int i = 0; i < vec / maxVec; i++) {
//       auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec),
//                           /*inbounds=*/true);
//       storeDShared(
//           rewriter, loc, newPtr, ctaId,
//           packLLVector(loc, ArrayRef(vals).slice(i * maxVec, maxVec), rewriter),
//           pred);
//     }
//     return;
//   }

//   // At this point we're committed to doing the store!
//   assert(elemBitwidth >= 8);
//   assert(elemTy.isInteger());
//   assert(1 <= vec && vec <= 4);
//   assert(vec * elemBitwidth <= 128);

//   // Get pointer to remote shared memory if needed.
//   if (ctaId.has_value()) {
//     ptr = mapa(rewriter, loc, ptr, *ctaId, pred);
//   }

//   PTXBuilder builder;
//   auto st = builder.create<>("st")
//                 ->o("shared::cta", ctaId.has_value())
//                 .o("shared", !ctaId.has_value())
//                 .v(vec, /*predicate=*/vec > 1)
//                 .b(elemBitwidth);
//   auto *ptrOpr = builder.newAddrOperand(ptr, "r");

//   PTXBuilder::Operand *valOpr;
//   std::string constraint = getConstraintForBitwidth(elemBitwidth);
//   if (vec > 1) {
//     SmallVector<std::pair<Value, std::string>> vecVals;
//     for (int i = 0; i < vec; i++) {
//       vecVals.push_back({b.extract_element(val, b.i32_val(i)), constraint});
//     }
//     valOpr = builder.newListOperand(vecVals);
//   } else {
//     valOpr = builder.newOperand(val, constraint);
//   }
//   st(ptrOpr, valOpr).predicate(pred, "b");
//   builder.launch(rewriter, loc, void_ty(ctx));
// }

// Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
//                               std::optional<Value> ctaId, Type loadTy,
//                               Value pred) const {
//   auto b = TritonLLVMOpBuilder(loc, rewriter);
//   MLIRContext *ctx = rewriter.getContext();
//   auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
//   assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

//   if (!isa<VectorType>(loadTy)) {
//     SmallVector<Value> values = unpackLLVector(
//         loc, loadDShared(rewriter, loc, ptr, ctaId, vec_ty(loadTy, 1), pred),
//         rewriter);
//     assert(values.size() == 1);
//     return values[0];
//   }

//   auto vecTy = cast<VectorType>(loadTy);
//   Type elemTy = vecTy.getElementType();
//   unsigned vec = vecTy.getNumElements();
//   unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
//   assert(llvm::isPowerOf2_32(vec));

//   if (elemBitwidth < 8) {
//     assert(vec == 1 &&
//            "don't know how to load/store vectors of sub-byte elems");
//     SmallVector<Value> vals = unpackLLVector(
//         loc, loadDShared(rewriter, loc, ptr, ctaId, int_ty(8), pred), rewriter);
//     assert(vals.size() == 1);
//     return b.bitcast(b.trunc(int_ty(elemBitwidth), vals[0]), elemTy);
//   }

//   // We only know how to load integers.
//   if (!elemTy.isInteger()) {
//     Type newLoadTy = vec_ty(int_ty(elemBitwidth), vec);
//     SmallVector<Value> vals = unpackLLVector(
//         loc, loadDShared(rewriter, loc, ptr, ctaId, newLoadTy, pred), rewriter);
//     for (Value &v : vals) {
//       v = b.bitcast(v, elemTy);
//     }
//     return packLLVector(loc, vals, rewriter);
//   }

//   // load/store ops only support v2 and v4.  If the vector width is larger than
//   // 4, we have two strategies for dealing with it.
//   //  1. If the element type is smaller than b32, load b32's instead.
//   //  2. Otherwise, split the load into multiple loads.
//   if (vec > 4 && elemBitwidth < 32) {
//     int newVec = vec / (32 / elemBitwidth);
//     auto newVecTy = vec_ty(i32_ty, newVec);
//     auto res = loadDShared(rewriter, loc, ptr, ctaId, newVecTy, pred);

//     // Unpack the b32's into the original vector type.
//     SmallVector<Value> vals;
//     for (Value v : unpackLLVector(loc, res, rewriter)) {
//       Value vv = b.bitcast(v, vec_ty(elemTy, 32 / elemBitwidth));
//       for (Value vvv : unpackLLVector(loc, vv, rewriter)) {
//         vals.push_back(vvv);
//       }
//     }
//     return packLLVector(loc, vals, rewriter);
//   }

//   if (vec * elemBitwidth > 128) {
//     assert(elemBitwidth == 32 || elemBitwidth == 64);
//     assert(llvm::isPowerOf2_32(vec));
//     int maxVec = 128 / elemBitwidth;

//     SmallVector<Value> vals;
//     for (int i = 0; i < vec / maxVec; i++) {
//       auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec),
//                           /*inbounds=*/true);
//       auto newVal = loadDShared(rewriter, loc, newPtr, ctaId,
//                                 vec_ty(elemTy, maxVec), pred);
//       for (Value v : unpackLLVector(loc, newVal, rewriter)) {
//         vals.push_back(v);
//       }
//     }
//     return packLLVector(loc, vals, rewriter);
//   }

//   // At this point we're committed to actually do the load!
//   assert(elemBitwidth >= 8);
//   assert(elemTy.isInteger());
//   assert(1 <= vec && vec <= 4);
//   assert(vec * elemBitwidth <= 128);

//   // Get pointer to remote shared memory if needed.
//   if (ctaId.has_value()) {
//     ptr = mapa(rewriter, loc, ptr, *ctaId, pred);
//   }

//   PTXBuilder builder;
//   auto ld = builder.create<>("ld")
//                 ->o("shared::cta", ctaId.has_value())
//                 .o("shared", !ctaId.has_value())
//                 .v(vec, /*predicate=*/vec > 1)
//                 .b(elemBitwidth);

//   Value load;
//   if (isConstantTruePred(pred)) {
//     Type resultTy = vec == 1 ? Type(int_ty(elemBitwidth))
//                              : Type(vec_ty(int_ty(elemBitwidth), vec));
//     load = b.load(resultTy, ptr);
//     if (vec > 1) {
//       Type structTy = struct_ty(SmallVector<Type>(vec, int_ty(elemBitwidth)));
//       Value structValue = b.undef(structTy);
//       for (int i = 0; i < vec; i++) {
//         structValue = b.insert_val(structTy, structValue,
//                                    b.extract_element(load, b.i32_val(i)), i);
//       }
//       load = structValue;
//     }
//   } else {
//     std::string elemConstraint = "=" + getConstraintForBitwidth(elemBitwidth);
//     auto *outOpr = vec == 1 ? builder.newOperand(elemConstraint)
//                             : builder.newListOperand(vec, elemConstraint);
//     ld(outOpr, builder.newAddrOperand(ptr, "r")).predicate(pred, "b");

//     Type resultTy =
//         vec == 1
//             ? Type(int_ty(elemBitwidth))
//             : Type(struct_ty(SmallVector<Type>(vec, int_ty(elemBitwidth))));
//     load = builder.launch(rewriter, loc, resultTy, /*hasSideEffects=*/true);
//   }
//   SmallVector<Value> resultVals = unpackLLElements(loc, load, rewriter);
//   return packLLVector(loc, resultVals, rewriter);
// }

// Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
//                              int i) const {
//   return LLVM::NVIDIA::shuffleXor(loc, rewriter, val, i);
// }

// Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
//                             int i) const {
//   return LLVM::NVIDIA::shuffleUp(loc, rewriter, val, i);
// }

// Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
//                              int i) const {
//   return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
// }

// Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
//                              Value i) const {
//   return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
// }

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::tts::NVIDIA::llGetPid(loc, rewriter, moduleOp, axis);
}


// // TODO (Keren): Currently, we have more restrictions than necessary when using
// // stmatrix.  These restrictions are retained from legacy code, and we could
// // relax some of them in the future.
// // TODO (Lezcano): The proper way of doing this is to directly try to fit the
// // relevant layout and return an std::optional<LinearLayout>. I'm keeping this
// // split to keep the current PR smaller
// bool TargetInfo::canUseStMatrix(RankedTensorType tensorTy,
//                                 ArrayRef<unsigned> repShape,
//                                 ArrayRef<unsigned> paddedRepShape,
//                                 ArrayRef<unsigned> order,
//                                 int swizzleByteSize) const {
//   if (computeCapability < 90) {
//     return false;
//   }
//   auto mmaLayout =
//       mlir::dyn_cast<NvidiaMmaEncodingAttr>(tensorTy.getEncoding());
//   if (!mmaLayout || !mmaLayout.isHopper())
//     return false;
//   if (isa<PointerType>(tensorTy.getElementType()))
//     return false;
//   if (tensorTy.getElementType().getIntOrFloatBitWidth() != 16)
//     return false;
//   if (order[0] != 1)
//     return false;

//   auto tensorShapePerCTA = getShapePerCTA(mmaLayout, tensorTy.getShape());
//   if (tensorShapePerCTA.size() != 2)
//     return false;
//   auto numIterations = ceil<unsigned>(tensorShapePerCTA[1], repShape[1]) *
//                        ceil<unsigned>(tensorShapePerCTA[0], repShape[0]);
//   if (numIterations > 1)
//     return false;
//   if (paddedRepShape[1] % 8 != 0)
//     return false;
//   if (swizzleByteSize != 0 && swizzleByteSize != 32 && swizzleByteSize != 64 &&
//       swizzleByteSize != 128)
//     return false;
//   return true;
// }

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__nv_umulhi" : "__nv_umul64hi";
  return funcName;
}

// void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
//                         int /*formatStrByteCount*/, ValueRange args) const {
//   auto *ctx = rewriter.getContext();
//   Type ptr = ptr_ty(ctx);
//   auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
//   auto funcOp = getVprintfDeclaration(rewriter);
//   auto loc = UnknownLoc::get(ctx);
//   auto b = TritonLLVMOpBuilder(loc, rewriter);

//   Value one = b.i32_val(1);
//   Value zero = b.i32_val(0);

//   Value bufferPtr = b.null(ptr);

//   SmallVector<Value, 16> newArgs;
//   if (args.size() >= 1) {
//     SmallVector<Type> argTypes;
//     for (auto arg : args) {
//       Type newType;
//       Value newArg;
//       std::tie(newType, newArg) = printfPromoteValue(rewriter, arg);
//       argTypes.push_back(newType);
//       newArgs.push_back(newArg);
//     }

//     Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
//     auto allocated =
//         rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(ctx), structTy, one,
//                                         /*alignment=*/0);

//     for (const auto &entry : llvm::enumerate(newArgs)) {
//       auto index = b.i32_val(entry.index());
//       auto fieldPtr =
//           b.gep(ptr_ty(ctx), structTy, allocated, ArrayRef<Value>{zero, index});
//       b.store(entry.value(), fieldPtr);
//     }
//     bufferPtr = b.bitcast(allocated, ptr);
//   }

//   SmallVector<Value> operands{formatStrStart, bufferPtr};
//   b.call(funcOp, operands);
// }

// void TargetInfo::printf(RewriterBase &rewriter, StringRef msg,
//                         ValueRange args) const {
//   assert(!msg.empty() && "printf with empty string not supported");
//   llvm::SmallString<64> msgNewline(msg);
//   msgNewline.push_back('\n');
//   msgNewline.push_back('\0');
//   Value msgValue =
//       LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
//                               "printfFormat_", msgNewline);
//   printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
// }

// void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
//                             StringRef message, StringRef file, StringRef func,
//                             int line) const {
//   auto b = TritonLLVMOpBuilder(loc, rewriter);
//   auto funcOp = getAssertfailDeclaration(rewriter);
//   auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
//   llvm::SmallString<64> messageString(message), fileString(file),
//       funcString(func);
//   messageString.push_back('\0');
//   fileString.push_back('\0');
//   funcString.push_back('\0');
//   Value messageStringVal =
//       LLVM::addStringToModule(loc, rewriter, "assertMessage_", messageString);
//   Value fileStringVal =
//       LLVM::addStringToModule(loc, rewriter, "assertFile_", fileString);
//   Value funcStringVal =
//       LLVM::addStringToModule(loc, rewriter, "assertFunc_", funcString);
//   Value lineNumber = b.i32_val(line);
//   Value charSize = b.int_val(sizeof(size_t) * 8, sizeof(char));
//   SmallVector<Value> operands = {messageStringVal, fileStringVal, lineNumber,
//                                  funcStringVal, charSize};
//   b.call(funcOp, operands);
// }

int TargetInfo::getSharedAddressSpace() const { return 3; }

bool TargetInfo::supportVectorizedAtomics() const {
  return computeCapability >= 90 && ptxVersion >= 81;
}

} // namespace mlir::triton::NVIDIA
