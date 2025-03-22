import triton
import torch
import triton.language as tl

@triton.autotune(
    configs = [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps = 4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages = 1),
    ],
    key = ["n_elements"],
)
@triton.jit
def int8_quantize_global(
    x_ptr, # Input
    abs_max_inv_ptr, # Pointer to scaling factor 1/abs(max_value)
    y_ptr, # Output
    n_elements, # Number of elements in input
    BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE  + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    abs_max_inv =  tl.load(abs_max_inv_ptr)

    y = tl.extra.cuda.libdevice.round(127.0 * (x*abs_max_inv)) # Rounding to nearest integer

    tl.store(y_ptr + offsets, y, mask = mask)

def quantize_global(x:torch.Tensor):
    abs_max = x.abs().max().unsqueeze(0) # [1,1]
    abs_max_inv = 1.0 / abs_max

    y = torch.empty(*x.shape, device = "cuda", dtype = torch.int8)

    assert y.is_cuda and x.is_cuda

    n_elements = x.numel()

    grid = lambda meta:(triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    int8_quantize_global[grid](
        x, abs_max_inv, y, n_elements
    )

    return y, abs_max


