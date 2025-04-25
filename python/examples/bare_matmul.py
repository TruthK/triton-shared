# this is a benchmark which multiplies square matrices with maximum block size
# to check the performance of tl.dot operation

import torch
import triton
import triton.language as tl
import benchmark


@triton.jit
def bare_matmul(X, Y, Z, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)  # block row id
    pid_y = tl.program_id(1)  # block column id

    offs_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offs_x[:, None] * K + offs_y[None, :])
    y = tl.load(Y + offs_x[:, None] * N + offs_y[None, :])

    z = tl.dot(x, y)

    tl.store(Z + offs_x[:, None] * N + offs_y[None, :], z)


@benchmark.measure()
def bench_matmul(N, provider):
    device = 'cuda'
    dtype = torch.float32
    a = torch.randn((N, N), device=device, dtype=dtype)
    b = torch.randn((N, N), device=device, dtype=dtype)
    c = torch.empty((N, N), device=device, dtype=dtype)
    grid = lambda META: (triton.cdiv(N, 32) * triton.cdiv(N, 32), )
    if provider == 'torch' or provider == 'test':
        c_ref = torch.matmul(a, b)
    if provider == 'triton' or provider == 'test':
        bare_matmul[grid](a, b, c, N, N, N, 32)
        if provider == 'test':
            torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=0)


if __name__ == "__main__":
    benchmark.select_kzx_backend()
    for provider in ['test', 'torch', 'triton']:
        bench_matmul(512, provider)
