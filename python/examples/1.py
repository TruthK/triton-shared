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

def fixed_triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.zeros((M, N), device='cuda')
    BLOCK = 32
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    mma[grid](a, b, c, M, N, K, BLOCK)
    return c

a = torch.randn(32, 32, device='cuda')
b = torch.randn(32, 32, device='cuda')
c_triton = fixed_triton_matmul(a, b)
print(c_triton)
