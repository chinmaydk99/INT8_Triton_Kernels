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
def quant_rowwise_kernel(
    input_ptr,
    output_ptr,
    output_maxs,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    NP2 : tl.constexpr):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    row_range = tl.arange(0, NP2)
    offsets = block_start + row_range

    mask = row_range < BLOCK_SIZE

    input = tl.load(input_ptr + offsets, mask = mask)

    abs_input = tl.abs(input)
    max_input = tl.max(tl.where(mask, abs_input, 0), axis = 0)

    output = tl.extra.cuda.libdevice.round(127*(abs_input/max_input))

    tl.store(output_ptr + offsets, output, mask = mask)
    tl.store(output_maxs + pid, max_input)

def quant_rowwise(x:torch.Tensor):
    output = torch.empty(*x.shape, device = "cuda", dtype = torch.int8)
    output_max = torch.empty(x.shape[0], device = "cuda", dtype = torch.int8) # Max value per row

    NP2 = triton.next_power_of_2(x.shape[1]) # Padded row size(number of columns) to next power of 2 for performance optimisations

    assert x.is_cuda and output.is_cuda

    n_elements = x.numel()
    grid = lambda meta: (x.shape[0],) # Quantization per row so we parallelize along the rows
    quant_rowwise_kernel[grid](
        x,
        output,
        output_max,
        n_elements,
        BLOCK_SIZE = x.shape[1],
        NP2 = NP2
    )

    return output, output_max

