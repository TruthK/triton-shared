import torch
import triton
import triton.language as tl

    
@triton.jit
def simple_matmul(X, Y, Z, M, N, K, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # 计算偏移量
    x_offs = row * BLOCK + tl.arange(0, BLOCK)
    y_offs = col * BLOCK + tl.arange(0, BLOCK)
    
    # 加载数据块
    x = tl.load(X + x_offs[:, None] * K + y_offs[None, :])
    y = tl.load(Y + x_offs[:, None] * N + y_offs[None, :])
    
    # 计算并存储结果
    tl.store(Z + x_offs[:, None] * N + y_offs[None, :], tl.dot(x, y))

# 封装函数
def triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device='cuda')
    BLOCK = 32
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    simple_matmul[grid](a, b, c, M, N, K, BLOCK)
    return c

# 测试
a = torch.randn(512, 512, device='cuda')
b = torch.randn(512, 512, device='cuda')
c = triton_matmul(a, b)
print(c)
