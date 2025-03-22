import triton
import torch
import triton.language as tl


@triton.autotune(
    configs=[
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 4}, num_warps=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 16}, num_warps=8),
  ],
  key = ["M", "N"]
)
@triton.jit
def int8_quantize_transpose_global(
    a_ptr,
    abs_max_inv_ptr,
    b_ptr,
    stride_am, stride_an,
    stride_bm, stride_bn,
    M, N,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    GROUP_SIZE_M : tl.constexpr
    ):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_groups = GROUP_SIZE_M * num_blocks_n
    group_id = pid // num_groups
    group_size_m = min(GROUP_SIZE_M, num_blocks_m - group_id * GROUP_SIZE_M)

    pid_m = group_id * GROUP_SIZE_M + pid % group_size_m
    pid_n = (pid // num_groups) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + (offs_m[:, None]*stride_am + offs_n[None,:] * stride_bn)
    mask = (offs_m[:, None] < M ) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs, mask = mask)
    abs_max_inv = tl.load(abs_max_inv_ptr)

    b_ptrs = b_ptr + (offs_m[:, None]*stride_bm + offs_n[None,:] * stride_bn) # Here the strides are defined in a way such that the rows in A are written as columns in b
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    output = tl.extra.cuda.libdevice.round(127.0 * (a*abs_max_inv))

    tl.store(b_ptrs, output, mask = mask)

def quantize_global_transpose(input):
    absmax = input.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    M, N = input.shape
    out = torch.empty(N, M, device="cuda", dtype=torch.int8)

    assert out.size(0) == N and out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert out.stride(0) == 1 or out.stride(1) == 1

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    int8_quantize_transpose_global[grid](
        input,
        absmax_inv,
        out,
        input.stride(0),
        input.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        N,
    )
    return out, absmax

M, N = 256, 256  # Matrix dimensions
input_tensor = torch.randn((M, N), device="cuda")
quantize_global_transpose(input_tensor)

