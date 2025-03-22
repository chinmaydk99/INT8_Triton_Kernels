

import math
import triton
import triton.language as tl
import torch

@triton.autotune(
        configs=[
            triton.Config({}, num_stages=1, num_warps=8),
            triton.Config({}, num_stages=2, num_warps=8),
            triton.Config({}, num_stages=4, num_warps=8),
            triton.Config({}, num_stages=8, num_warps=8),
            triton.Config({}, num_stages=1),
            triton.Config({}, num_stages=2),
            triton.Config({}, num_stages=4),
            triton.Config({}, num_stages=8),
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=["n_elements"],
    )
@triton.jit
def quant_columnwise_kernel(input_ptr,
                            output_ptr,
                            col_max_ptr,
                            n_elements,
                            M : tl.constexpr,
                            N : tl.constexpr,
                            BLOCK_SIZE : tl.constexpr,
                            NP2 : tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid # For row wise this would be pid*BLOCK_SIZE
    arange = tl.arange(0, NP2)
    offsets = block_start  + arange * N # row_idx * num_columns + col_idx
    mask = arange < M

    input = tl.load(input_ptr + offsets, mask=mask)
    inp_abs = tl.abs(input)
    col_max = tl.max(tl.where(mask, inp_abs, 0), axis = 0)

    output = tl.extra.cuda.libdevice.round(127.0 * (input / col_max))

    new_start = pid*M
    new_offset = arange + new_start # Would be storing it at r_Idx*n_cols + c_idx. Its transposed so storing it at c_idx * n_rows + r_iidx

    tl.store(output_ptr + new_offset, output, mask=mask)
    tl.store(col_max_ptr + pid , col_max)

def quant_columnwise_transpose(x:torch.Tensor):
    M, N = x.shape
    output = torch.empty(N, M, dtype=torch.int8, device=x.device)
    col_maxs = torch.empty(x.shape[1], dtype=torch.float16, device=x.device)

    NP2 = triton.next_power_of_2(M)

    assert x.is_cuda and output.is_cuda
    n_elements = x.numel()

    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = lambda meta: (N,)
    quant_columnwise_kernel[grid](x,
                                  output,
                                  col_maxs,
                                  n_elements,
                                  M = M,
                                  N =  N,
                                  BLOCK_SIZE = M,
                                  NP2 = NP2)

    return output, col_maxs

M, N = 256, 128
input_tensor = torch.randn((M, N), device="cuda")


quantized_output, col_maxs = quant_columnwise_transpose(input_tensor)
col_maxs = col_maxs.unsqueeze(0)
quantized_output = torch.transpose(quantized_output, 0, 1)

quantized_output.shape, col_maxs.shape

expected_maxs = input_tensor.abs().max(dim=0, keepdim=True)[0]
expected_output = (127.0 * (input_tensor / expected_maxs)).round().to(torch.int8)

expected_maxs = expected_maxs.to(torch.float16)

expected_output.shape, expected_maxs.shape

if torch.allclose(quantized_output, expected_output) and torch.allclose(col_maxs, expected_maxs.squeeze(0)):
    print("✅ Triton column-wise kernel matches PyTorch output!")
else:
    print("❌ Mismatch detected!")
    print("Max absolute difference:", (quantized_output - expected_output).abs().max())

