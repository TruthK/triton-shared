import torch
import triton
import triton.language as tl

@triton.jit
def mma(X, Y, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_offsets = pid_m * BLOCK + tl.arange(0, BLOCK)
    col_offsets = pid_n * BLOCK + tl.arange(0, BLOCK)
    
    x_ptrs = X + row_offsets[:, None] * K + tl.arange(0, K)[None, :]
    y_ptrs = Y + tl.arange(0, K)[:, None] * N + col_offsets[None, :]
    
    x = tl.load(x_ptrs)
    y = tl.load(y_ptrs)
    
    c = tl.dot(x, y)
    
    output_ptrs = Z + row_offsets[:, None] * N + col_offsets[None, :]
    tl.store(output_ptrs, c)

def triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.zeros((M, N), device='cuda')
    BLOCK = 64
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    mma[grid](a, b, c, M, N, K, BLOCK)
    return c


# %%
# Benchmark
# ---------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 矩阵的行列大小
        x_vals=[32, 64, 128, 256, 512],  # 不同的方阵大小
        x_log=True,  # 对数刻度
        line_arg='provider',  # 区分 Triton 和 Torch
        line_vals=['tts', 'torch'],
        line_names=['tts', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='TFLOPS',  # 性能单位
        plot_name='matmul-performance',  # 保存图片名称
        args={},
    )
)
def benchmark(size, provider):
    # 生成随机输入
    a = torch.randn((size, size), device='cuda')
    b = torch.randn((size, size), device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    # 根据 provider 选择函数
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b), quantiles=quantiles)
    # 计算 TFLOPS: 2*M*N*K 次浮点运算
    perf = lambda ms: 2 * size * size * size * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)
