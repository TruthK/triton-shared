#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define WIDTH 64
#define HEIGHT 64
#define K 64

const char* ptxCode = R"( 
    
.version 8.4
.target sm_75
.address_size 64

	// .globl	bare_matmul
.extern .shared .align 16 .b8 __dynamic_shared_memory__[];

.visible .entry bare_matmul(
	.param .u64 .ptr .global .align 1 bare_matmul_param_0,
	.param .u64 .ptr .global .align 1 bare_matmul_param_1,
	.param .u64 .ptr .global .align 1 bare_matmul_param_2,
	.param .u32 bare_matmul_param_3,
	.param .u32 bare_matmul_param_4,
	.param .u64 .ptr .global .align 1 bare_matmul_param_5
)
.reqntid 128, 1, 1
{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<50>;
	.reg .f32 	%f<57>;
	.reg .b64 	%rd<154>;

	ld.param.u64 	%rd27, [bare_matmul_param_0];
	ld.param.u64 	%rd28, [bare_matmul_param_1];
	mov.u32 	%r2, %tid.x;
	cvt.u64.u32 	%rd29, %r2;
	ld.param.u32 	%r3, [bare_matmul_param_3];
	mov.u32 	%r4, %ctaid.x;
	ld.param.u32 	%r5, [bare_matmul_param_4];
	mov.u32 	%r6, %ctaid.y;
	shl.b32 	%r7, %r4, 5;
	shl.b32 	%r8, %r6, 5;
	cvt.u64.u32 	%rd30, %r8;
	cvt.s64.s32 	%rd31, %r5;
	mul.wide.s32 	%rd32, %r5, %r7;
	cvt.s64.s32 	%rd1, %r3;
	mul.wide.s32 	%rd33, %r3, %r7;
	add.s64 	%rd2, %rd33, %rd30;
	mov.u32 	%r1, %tid.y;
	shr.u64 	%rd34, %rd29, 2;
	and.b64  	%rd35, %rd34, 24;
	mul.wide.u32 	%rd36, %r2, 4;
	shr.u64 	%rd37, %rd29, 1;
	xor.b64  	%rd38, %rd37, %rd36;
	and.b64  	%rd39, %rd38, 12;
	shr.u64 	%rd3, %rd29, 3;
	and.b64  	%rd4, %rd36, 28;
	xor.b64  	%rd40, %rd3, %rd29;
	shl.b64 	%rd41, %rd40, 2;
	and.b64  	%rd5, %rd41, 28;
	shl.b64 	%rd42, %rd32, 2;
	add.s64 	%rd43, %rd27, %rd42;
	mul.wide.u32 	%rd44, %r8, 4;
	add.s64 	%rd45, %rd43, %rd44;
	mul.lo.s64 	%rd46, %rd34, %rd31;
	shl.b64 	%rd47, %rd46, 2;
	add.s64 	%rd48, %rd45, %rd47;
	shl.b64 	%rd49, %rd2, 2;
	add.s64 	%rd50, %rd28, %rd49;
	shl.b64 	%rd51, %rd4, 2;
	add.s64 	%rd6, %rd50, %rd51;
	setp.ne.s32 	%p1, %r1, 0;
	mov.u32 	%r9, %laneid;
	cvt.u64.u32 	%rd52, %r9;
	shr.u64 	%rd53, %rd52, 2;
	and.b64  	%rd54, %rd53, 4;
	mul.wide.u32 	%rd55, %r9, 2;
	and.b64  	%rd56, %rd55, 12;
	xor.b64  	%rd57, %rd54, %rd56;
	mul.wide.u32 	%rd58, %r9, 16;
	and.b64  	%rd59, %rd58, 240;
	or.b64  	%rd7, %rd59, %rd57;
	or.b64  	%rd60, %rd54, 8;
	xor.b64  	%rd61, %rd60, %rd56;
	or.b64  	%rd8, %rd59, %rd61;
	or.b64  	%rd62, %rd58, 256;
	or.b64  	%rd9, %rd62, %rd57;
	or.b64  	%rd10, %rd62, %rd61;
	shl.b64 	%rd63, %rd53, 3;
	sub.s64 	%rd64, %rd55, %rd63;
	shl.b64 	%rd65, %rd53, 7;
	mov.u64 	%rd66, __dynamic_shared_memory__;
	add.s64 	%rd67, %rd66, %rd65;
	add.s64 	%rd68, %rd64, %rd35;
	shl.b64 	%rd69, %rd68, 2;
	add.s64 	%rd70, %rd67, %rd69;
	and.b64  	%rd71, %rd52, 3;
	or.b64  	%rd11, %rd71, 4;
	shl.b64 	%rd72, %rd71, 2;
	shl.b64 	%rd73, %rd11, 2;
	shl.b64 	%rd12, %rd71, 5;
	or.b64  	%rd74, %rd72, 16;
	add.s64 	%rd13, %rd70, 16384;
	or.b64  	%rd75, %rd35, %rd53;
	xor.b64  	%rd14, %rd75, %rd72;
	xor.b64  	%rd15, %rd75, %rd73;
	or.b64  	%rd16, %rd12, %rd14;
	xor.b64  	%rd76, %rd75, %rd74;
	or.b64  	%rd18, %rd76, %rd12;
	bar.sync 	0;
	shl.b64 	%rd77, %rd39, 2;
	shl.b64 	%rd78, %rd34, 6;
	or.b64  	%rd79, %rd78, %rd77;
	add.s64 	%rd80, %rd66, %rd79;
	add.s64 	%rd20, %rd80, 8192;
	shl.b64 	%rd81, %rd36, 2;
	and.b64  	%rd82, %rd81, 48;
	add.s64 	%rd21, %rd48, %rd82;
	cp.async.cg.shared.global [%rd20], [%rd21], 16;
	shl.b64 	%rd22, %rd3, 5;
	or.b64  	%rd83, %rd22, %rd5;
	shl.b64 	%rd84, %rd83, 2;
	add.s64 	%rd85, %rd66, %rd84;
	mul.lo.s64 	%rd23, %rd3, %rd1;
	shl.b64 	%rd86, %rd23, 2;
	add.s64 	%rd87, %rd6, %rd86;
	cp.async.cg.shared.global [%rd85], [%rd87], 16;
	cp.async.commit_group;
	cp.async.wait_group 0;
	bar.sync 	0;
	@%p1 bra 	$L__BB0_2;
	or.b64  	%rd17, %rd16, 256;
	or.b64  	%rd19, %rd18, 384;
	shl.b64 	%rd88, %rd7, 2;
	add.s64 	%rd90, %rd66, 8192;
	add.s64 	%rd91, %rd90, %rd88;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r10, %r11, %r12, %r13}, [%rd91];
	shl.b64 	%rd92, %rd8, 2;
	add.s64 	%rd93, %rd90, %rd92;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r14, %r15, %r16, %r17}, [%rd93];
	shl.b64 	%rd94, %rd9, 2;
	add.s64 	%rd95, %rd90, %rd94;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r18, %r19, %r20, %r21}, [%rd95];
	shl.b64 	%rd96, %rd10, 2;
	add.s64 	%rd97, %rd90, %rd96;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r22, %r23, %r24, %r25}, [%rd97];
	shl.b64 	%rd98, %rd16, 2;
	add.s64 	%rd99, %rd66, %rd98;
	ld.shared.u32 	%r26, [%rd99];
	shl.b64 	%rd100, %rd15, 2;
	shl.b64 	%rd101, %rd11, 7;
	or.b64  	%rd102, %rd101, %rd100;
	add.s64 	%rd103, %rd66, %rd102;
	ld.shared.u32 	%r27, [%rd103];
	shl.b64 	%rd104, %rd17, 2;
	add.s64 	%rd105, %rd66, %rd104;
	ld.shared.u32 	%r28, [%rd105];
	shl.b64 	%rd106, %rd19, 2;
	add.s64 	%rd107, %rd66, %rd106;
	ld.shared.u32 	%r29, [%rd107];
	ld.shared.v2.f32 	{%f1, %f2}, [%rd13];
	ld.shared.v2.f32 	{%f3, %f4}, [%rd13+1024];
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%f5, %f6, %f7, %f8},
		{%r10, %r11, %r12, %r13},
		{%r26, %r27},
		{%f1, %f2, %f3, %f4};
	ld.shared.v2.f32 	{%f9, %f10}, [%rd13+2048];
	ld.shared.v2.f32 	{%f11, %f12}, [%rd13+3072];
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%f13, %f14, %f15, %f16},
		{%r18, %r19, %r20, %r21},
		{%r26, %r27},
		{%f9, %f10, %f11, %f12};
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%f17, %f18, %f19, %f20},
		{%r14, %r15, %r16, %r17},
		{%r28, %r29},
		{%f5, %f6, %f7, %f8};
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%f21, %f22, %f23, %f24},
		{%r22, %r23, %r24, %r25},
		{%r28, %r29},
		{%f13, %f14, %f15, %f16};
	st.shared.v2.f32 	[%rd13], {%f17, %f18};
	st.shared.v2.f32 	[%rd13+1024], {%f19, %f20};
	st.shared.v2.f32 	[%rd13+2048], {%f21, %f22};
	st.shared.v2.f32 	[%rd13+3072], {%f23, %f24};
$L__BB0_2:
	ld.param.u64 	%rd26, [bare_matmul_param_2];
	bar.sync 	0;
	add.s64 	%rd108, %rd20, 2048;
	add.s64 	%rd109, %rd21, 64;
	cp.async.cg.shared.global [%rd108], [%rd109], 16;
	or.b64  	%rd24, %rd22, 512;
	or.b64  	%rd111, %rd24, %rd5;
	shl.b64 	%rd112, %rd111, 2;
	add.s64 	%rd114, %rd66, %rd112;
	shl.b64 	%rd25, %rd1, 4;
	add.s64 	%rd115, %rd23, %rd25;
	shl.b64 	%rd116, %rd115, 2;
	add.s64 	%rd117, %rd6, %rd116;
	cp.async.cg.shared.global [%rd114], [%rd117], 16;
	cp.async.commit_group;
	cp.async.wait_group 0;
	bar.sync 	0;
	@%p1 bra 	$L__BB0_4;
	shl.b64 	%rd118, %rd7, 2;
	add.s64 	%rd120, %rd66, 8192;
	add.s64 	%rd121, %rd120, %rd118;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r30, %r31, %r32, %r33}, [%rd121+2048];
	shl.b64 	%rd122, %rd8, 2;
	add.s64 	%rd123, %rd120, %rd122;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r34, %r35, %r36, %r37}, [%rd123+2048];
	shl.b64 	%rd124, %rd9, 2;
	add.s64 	%rd125, %rd120, %rd124;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r38, %r39, %r40, %r41}, [%rd125+2048];
	shl.b64 	%rd126, %rd10, 2;
	add.s64 	%rd127, %rd120, %rd126;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r42, %r43, %r44, %r45}, [%rd127+2048];
	add.s64 	%rd128, %rd14, %rd12;
	shl.b64 	%rd129, %rd128, 2;
	add.s64 	%rd130, %rd66, %rd129;
	ld.shared.u32 	%r46, [%rd130+2048];
	add.s64 	%rd131, %rd15, %rd12;
	shl.b64 	%rd132, %rd131, 2;
	add.s64 	%rd133, %rd66, %rd132;
	ld.shared.u32 	%r47, [%rd133+2560];
	shl.b64 	%rd134, %rd16, 2;
	add.s64 	%rd135, %rd66, %rd134;
	ld.shared.u32 	%r48, [%rd135+3072];
	shl.b64 	%rd136, %rd18, 2;
	add.s64 	%rd137, %rd66, %rd136;
	ld.shared.u32 	%r49, [%rd137+3584];
	ld.shared.v2.f32 	{%f25, %f26}, [%rd13];
	ld.shared.v2.f32 	{%f27, %f28}, [%rd13+1024];
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%f29, %f30, %f31, %f32},
		{%r30, %r31, %r32, %r33},
		{%r46, %r47},
		{%f25, %f26, %f27, %f28};
	ld.shared.v2.f32 	{%f33, %f34}, [%rd13+2048];
	ld.shared.v2.f32 	{%f35, %f36}, [%rd13+3072];
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%f37, %f38, %f39, %f40},
		{%r38, %r39, %r40, %r41},
		{%r46, %r47},
		{%f33, %f34, %f35, %f36};
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%f41, %f42, %f43, %f44},
		{%r34, %r35, %r36, %r37},
		{%r48, %r49},
		{%f29, %f30, %f31, %f32};
	mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
		{%f45, %f46, %f47, %f48},
		{%r42, %r43, %r44, %r45},
		{%r48, %r49},
		{%f37, %f38, %f39, %f40};
	st.shared.v2.f32 	[%rd13], {%f41, %f42};
	st.shared.v2.f32 	[%rd13+1024], {%f43, %f44};
	st.shared.v2.f32 	[%rd13+2048], {%f45, %f46};
	st.shared.v2.f32 	[%rd13+3072], {%f47, %f48};
$L__BB0_4:
	or.b64  	%rd138, %rd22, %rd4;
	shl.b64 	%rd139, %rd138, 2;
	add.s64 	%rd141, %rd66, 16384;
	add.s64 	%rd142, %rd141, %rd139;
	ld.shared.v4.f32 	{%f49, %f50, %f51, %f52}, [%rd142];
	add.s64 	%rd144, %rd26, %rd49;
	add.s64 	%rd146, %rd144, %rd86;
	add.s64 	%rd148, %rd146, %rd51;
	st.global.f32 	[%rd148+12], %f52;
	st.global.f32 	[%rd148+8], %f51;
	st.global.f32 	[%rd148+4], %f50;
	st.global.f32 	[%rd148], %f49;
	or.b64  	%rd149, %rd24, %rd4;
	shl.b64 	%rd150, %rd149, 2;
	add.s64 	%rd151, %rd141, %rd150;
	ld.shared.v4.f32 	{%f53, %f54, %f55, %f56}, [%rd151];
	shl.b64 	%rd152, %rd25, 2;
	add.s64 	%rd153, %rd148, %rd152;
	st.global.f32 	[%rd153+12], %f56;
	st.global.f32 	[%rd153+8], %f55;
	st.global.f32 	[%rd153+4], %f54;
	st.global.f32 	[%rd153], %f53;
	ret;

}

)";

void loadPTXAndLaunch(float* A, float* B, float* C, int M, int N, int K) {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction kernel;

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDA_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    CUDA_CHECK(cuModuleLoadDataEx(&cuModule, ptxCode, 0, nullptr, nullptr));
    CUDA_CHECK(cuModuleGetFunction(&kernel, cuModule, "bare_matmul"));

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    void* args[] = {
        &d_A, &d_B, &d_C,
        &K, &N, nullptr  // 最后一个参数通常是 Workspace (可传 nullptr)
    };

    // 配置线程和共享内存
    int threadsPerBlock = 128;
    int sharedMemBytes = 16384 + 4096;  // 粗略估算: 确保足够
    dim3 grid(N / 32, M / 32);
    dim3 block(threadsPerBlock);

    CUDA_CHECK(cuLaunchKernel(kernel,
        grid.x, grid.y, 1,
        block.x, 1, 1,
        sharedMemBytes, 0,
        args, 0));

    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cuModuleUnload(cuModule));
    CUDA_CHECK(cuCtxDestroy(cuContext));
}

int main() {
    const int M = HEIGHT;
    const int N = WIDTH;
    const int Kval = K;

    std::vector<float> A(M * Kval, 1.0f);
    std::vector<float> B(Kval * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    loadPTXAndLaunch(A.data(), B.data(), C.data(), M, N, Kval);

    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
