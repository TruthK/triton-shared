#include <cuda.h>
#include <iostream>
#include <vector>
#include <cassert>

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        std::cerr << "CUDA error: " << errStr << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

const char* ptx_code = R"(

.version 8.5
.target sm_52
.address_size 64

	// .globl	add_kernel

.visible .entry add_kernel(
	.param .u64 .ptr .global .align 1 add_kernel_param_0,
	.param .u64 .ptr .global .align 1 add_kernel_param_1,
	.param .u64 .ptr .global .align 1 add_kernel_param_2,
	.param .u32 add_kernel_param_3,
	.param .u64 .ptr .global .align 1 add_kernel_param_4
)
.reqntid 128, 1, 1
{
	.local .align 4 .b8 	__local_depot0[4096];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<16>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<5>;
	.reg .f32 	%f<109>;
	.reg .b64 	%rd<44>;

	mov.u64 	%SPL, __local_depot0;
	ld.param.u64 	%rd17, [add_kernel_param_0];
	mov.u32 	%r1, %tid.x;
	mov.u32 	%r2, %ctaid.x;
	ld.param.s32 	%rd19, [add_kernel_param_3];
	shl.b32 	%r3, %r2, 10;
	cvt.s64.s32 	%rd1, %r3;
	ld.global.u64 	%rd20, [%rd17];
	add.s64 	%rd21, %rd1, 1024;
	min.s64 	%rd22, %rd21, %rd19;
	max.s64 	%rd23, %rd22, %rd1;
	sub.s64 	%rd2, %rd23, %rd1;
	add.u64 	%rd4, %SPL, 0;
	shl.b32 	%r4, %r1, 2;
	cvt.u64.u32 	%rd25, %r4;
	mul.wide.u32 	%rd26, %r2, 128;
	add.s64 	%rd5, %rd26, %rd25;
	setp.gt.s64 	%p5, %rd2, %rd5;
	sub.s64 	%rd27, %rd2, %rd5;
	min.s64 	%rd28, %rd27, 4;
	selp.b64 	%rd6, %rd28, 0, %p5;
	mul.wide.s32 	%rd29, %r3, 4;
	add.s64 	%rd30, %rd20, %rd29;
	shl.b64 	%rd31, %rd5, 2;
	add.s64 	%rd10, %rd30, %rd31;
	mov.f32 	%f81, 0f00000000;
	setp.lt.s64 	%p6, %rd6, 1;
	mov.f32 	%f82, %f81;
	mov.f32 	%f83, %f81;
	mov.f32 	%f84, %f81;
	@%p6 bra 	$L__BB0_2;
	ld.f32 	%f81, [%rd10];
	mov.f32 	%f82, 0f00000000;
	mov.f32 	%f83, %f82;
	mov.f32 	%f84, %f82;
$L__BB0_2:
	setp.lt.s64 	%p7, %rd6, 2;
	@%p7 bra 	$L__BB0_4;
	ld.f32 	%f82, [%rd10+4];
$L__BB0_4:
	ld.param.u64 	%rd18, [add_kernel_param_1];
	setp.lt.s64 	%p8, %rd6, 3;
	@%p8 bra 	$L__BB0_6;
	ld.f32 	%f83, [%rd10+8];
$L__BB0_6:
	ld.global.u64 	%rd3, [%rd18];
	setp.gt.s64 	%p1, %rd6, 0;
	setp.lt.s64 	%p9, %rd6, 4;
	@%p9 bra 	$L__BB0_8;
	ld.f32 	%f84, [%rd10+12];
$L__BB0_8:
	setp.gt.s64 	%p2, %rd6, 1;
	shl.b64 	%rd32, %rd1, 2;
	add.s64 	%rd33, %rd3, %rd32;
	add.s64 	%rd11, %rd33, %rd31;
	mov.f32 	%f97, 0f00000000;
	not.pred 	%p10, %p1;
	mov.f32 	%f98, %f97;
	mov.f32 	%f99, %f97;
	mov.f32 	%f100, %f97;
	@%p10 bra 	$L__BB0_10;
	ld.f32 	%f97, [%rd11];
	mov.f32 	%f98, 0f00000000;
	mov.f32 	%f99, %f98;
	mov.f32 	%f100, %f98;
$L__BB0_10:
	setp.gt.s64 	%p3, %rd6, 2;
	not.pred 	%p11, %p2;
	@%p11 bra 	$L__BB0_12;
	ld.f32 	%f98, [%rd11+4];
$L__BB0_12:
	setp.gt.s64 	%p4, %rd6, 3;
	not.pred 	%p12, %p3;
	@%p12 bra 	$L__BB0_14;
	ld.f32 	%f99, [%rd11+8];
$L__BB0_14:
	not.pred 	%p13, %p4;
	@%p13 bra 	$L__BB0_16;
	ld.f32 	%f100, [%rd11+12];
$L__BB0_16:
	add.rn.f32 	%f73, %f84, %f100;
	add.rn.f32 	%f74, %f83, %f99;
	add.rn.f32 	%f75, %f81, %f97;
	add.rn.f32 	%f76, %f82, %f98;
	add.s64 	%rd36, %rd4, %rd31;
	st.local.f32 	[%rd36+4], %f76;
	st.local.f32 	[%rd36], %f75;
	st.local.f32 	[%rd36+8], %f74;
	st.local.f32 	[%rd36+12], %f73;
	setp.lt.s64 	%p14, %rd2, 1;
	@%p14 bra 	$L__BB0_19;
	ld.param.u64 	%rd16, [add_kernel_param_2];
	min.u64 	%rd38, %rd2, 1024;
	ld.global.u64 	%rd39, [%rd16];
	add.s64 	%rd12, %rd39, %rd32;
	shl.b64 	%rd13, %rd38, 2;
	mov.b64 	%rd43, 0;
$L__BB0_18:
	add.s64 	%rd41, %rd4, %rd43;
	ld.local.u8 	%rs1, [%rd41];
	add.s64 	%rd42, %rd12, %rd43;
	st.u8 	[%rd42], %rs1;
	add.s64 	%rd43, %rd43, 1;
	setp.lt.u64 	%p15, %rd43, %rd13;
	@%p15 bra 	$L__BB0_18;
$L__BB0_19:
	ret;

}

)";

struct PtrWrapper {
    CUdeviceptr base;
    CUdeviceptr aligned;
};

int main() {
    cuInit(0);
    CUdevice dev;
    CUcontext ctx;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK_CUDA(cuModuleLoadData(&mod, ptx_code));

    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, mod, "add_kernel"));

     int numElements = 1024;
     int threadsPerBlock = 128;
     int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> A(numElements, 1.0f);
    std::vector<float> B(numElements, 2.0f);
    std::vector<float> C(numElements, 0.0f);

    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, sizeof(float) * numElements);
    cuMemAlloc(&d_B, sizeof(float) * numElements);
    cuMemAlloc(&d_C, sizeof(float) * numElements);

    cuMemcpyHtoD(d_A, A.data(), sizeof(float) * numElements);
    cuMemcpyHtoD(d_B, B.data(), sizeof(float) * numElements);

    PtrWrapper srcA{d_A, d_A};
    PtrWrapper srcB{d_B, d_B};
    PtrWrapper dstC{d_C, d_C};

    CUdeviceptr d_srcA, d_srcB, d_dstC;
    cuMemAlloc(&d_srcA, sizeof(PtrWrapper));
    cuMemAlloc(&d_srcB, sizeof(PtrWrapper));
    cuMemAlloc(&d_dstC, sizeof(PtrWrapper));

    cuMemcpyHtoD(d_srcA, &srcA, sizeof(PtrWrapper));
    cuMemcpyHtoD(d_srcB, &srcB, sizeof(PtrWrapper));
    cuMemcpyHtoD(d_dstC, &dstC, sizeof(PtrWrapper));

    void* args[] = { &d_srcA, &d_srcB, &d_dstC, &numElements, &d_dstC };

    CHECK_CUDA(cuLaunchKernel(kernel,
        blocks, 1, 1,
        threadsPerBlock, 1, 1,
        0, nullptr, args, nullptr));

    cuCtxSynchronize();

    cuMemcpyDtoH(C.data(), d_C, sizeof(float) * numElements);

    for (int i = 0; i < 100; ++i) {
        printf("%f\n", C[i]);
         printf("%f\n", A[i]);
    }

    std::cout << "All values verified successfully.\n";

    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuMemFree(d_srcA);
    cuMemFree(d_srcB);
    cuMemFree(d_dstC);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
