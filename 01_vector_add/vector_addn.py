import triton
import torch
import triton.language as tl

device = triton.runtime.driver.active.get_active_torch_device()
@triton.jit
def add_kernel(x_ptr, y_ptr, op_ptr, n_elem, block_size: tl.constexpr):

    pid = tl.program_id(0)
    block_start = pid * block_size
    offset = block_start + tl.arange(0, block_size)

    mask = offset < n_elem
    x = tl.load(x_ptr + offset, mask = mask)
    y = tl.load(y_ptr + offset, mask = mask)

    output = x + y
    tl.store(op_ptr + offset, output, mask= mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    # assert x.is_cuda and y.is_cuda
    assert x.device == y.device == output.device == device
    n_elem = output.numel()

    grid = lambda meta: (triton.cdiv(n_elem, meta['block_size']), )
    add_kernel[grid](x, y, output, n_elem, block_size = 1024)

    return output

torch.manual_seed(42)
size = 1024
x = torch.randn(size, device = device)
y = torch.randn(size, device= device)
torch_output = x + y
triton_output = add(x, y)
print(torch_output)
print(triton_output)

print(f'difference: {torch_output - triton_output}')
