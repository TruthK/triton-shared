#ifndef TTS_CODEGEN_DIALECT_CODEGEN_UTILS_H_
#define TTS_CODEGEN_DIALECT_CODEGEN_UTILS_H_

#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenInterfaces.h"
#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenTypes.h"
#include "triton-shared/Dialect/Encoding/IR/EncodingOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tts::GPU {

bool operator==(TileSwizzle::Dim lhs, TileSwizzle::Dim rhs);
bool operator!=(TileSwizzle::Dim lhs, TileSwizzle::Dim rhs);
bool operator==(const TileSwizzle &lhs, const TileSwizzle &rhs);
bool operator!=(const TileSwizzle &lhs, const TileSwizzle &rhs);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              TileSwizzle::Dim::Kind kind);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TileSwizzle::Dim dim);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TileSwizzle &swizzle);

bool operator==(const MaterializeEncodingInfo &lhs,
                const MaterializeEncodingInfo &rhs);
bool operator!=(const MaterializeEncodingInfo &lhs,
                const MaterializeEncodingInfo &rhs);

std::string convertSwizzleKindToString(TileSwizzle::Dim::Kind kind);
std::optional<TileSwizzle::Dim::Kind> convertStringToSwizzleKind(StringRef str);

DictionaryAttr serializeTileSwizzle(MLIRContext *ctx,
                                    const TileSwizzle &swizzle);
std::optional<TileSwizzle> deserializeTileSwizzle(DictionaryAttr attr);

DictionaryAttr serializeEncodingInfo(MLIRContext *ctx,
                                     const MaterializeEncodingInfo &info);
std::optional<MaterializeEncodingInfo>
deserializeEncodingInfo(DictionaryAttr attr);

bool isIdentityLayout(const MaterializeEncodingInfo &info);

SmallVector<int64_t>
getExpandedTileShape(const TileSwizzle::ExpandShapeType &expandShape);

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

MaterializeEncodingInfo
getEncodingInfoForMatmul(Encoding::EncodingAttr encoding, TileMxNxK tileMxNxK);

} // namespace mlir::tts::GPU

#endif // TTS_CODEGEN_DIALECT_CODEGEN_UTILS_H_
