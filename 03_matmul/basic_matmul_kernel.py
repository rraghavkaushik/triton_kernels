'''My Note: So the matmul operation I am trying to implement
C = A x B.
A - (M x K)
B - (K x N)
C - (M x N)
So, each elem C[i, j] will be ∑_k A[i,k] * B[k, j]

Note: 
- Unless all operations are forced to float32, the tl.dot operations uses mixed precision even though the native dtype for A100s is Float32. 
'''

import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_k = tl.arange(0, BLOCK_SIZE)

    A_ptrs = A_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    B_ptrs = B_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
    C_ptrs = C_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype = tl.float32)
    for k in range(0, K, BLOCK_SIZE):
`       '''either cast inputs to float32 explicitly, or force to accumulate in that dtype'''

        a = tl.load(A_ptrs)
        b = tl.load(B_ptrs)
        acc += tl.dot(a, b, input_precision = "ieee")

        '''
        a = tl.load(A_ptrs, dtype = torch.flaot32)
        b = tl.load(B_ptrs, dtype = torch.float32)
        acc = tl.dot(a, b) '''
        A_ptrs += BLOCK_SIZE * stride_ak
        B_ptrs += BLOCK_SIZE * stride_bk

    tl.store(C_ptrs, acc)

import torch
def matmul(A, B):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M,N), device = 'cuda', dtype = torch.float32)

    grid = ((M + 31 ) // 32, (N + 31) // 32)
    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE = 32) 
    return C

device = 'cuda'
A = torch.randn((128, 128), device = device)
B = torch.randn((128, 128), device = device)
C = matmul(A, B)
C_ = torch.matmul(A, B)

print(torch.allclose(C, C_, atol = 1e-2))
# print(C, C_)
