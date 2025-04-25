#include <stdio.h>
#include <inttypes.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <string.h>

void print_i32(int32_t x) {
    printf("MLIR print_i32 says: %" PRId32 "\n", x);
}
void print_f32(float x) {
    printf("MLIR print_f32 says: %f\n", x);
}


// Signature: (allocated_ptr, aligned_ptr, offset, sizes[2], strides[2])
// Total: 7 arguments
void print_memref_f32(float *allocated, float *aligned, int64_t offset,
                      int64_t size0, int64_t size1,
                      int64_t stride0, int64_t stride1) {
    printf("Matrix [%ld x %ld] (offset=%ld):\n", size0, size1, offset);
    for (int64_t i = 0; i < size0; ++i) {
        for (int64_t j = 0; j < size1; ++j) {
            float val = *(aligned + offset + i * stride0 + j * stride1);
            printf("%6.2f ", val);
        }
        printf("\n");
    }
}

void memrefCopy(int64_t size, void *src, void *dst) {
    memcpy(dst, src, size);
}

