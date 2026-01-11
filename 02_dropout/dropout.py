import triton
import torch
import triton.language as tl

device = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _dropout_kernel(x_ptr, x_keep_ptr, op_ptr, n_elem, p, block_size: tl.constexpr):

  pid = tl.program_id(axis = 0)
  block_start = pid * block_size
  offset = block_start + tl.arange(0, block_size)

  mask = offset < n_elem
  x = tl.load(x_ptr + offset, mask = mask)
  x_keep = tl.load(x_keep_ptr + offset, mask = mask)
  output = tl.where(x_keep, x / (1- p), 0.0)

  tl.store(op_ptr + offset, output, mask = mask)


def dropout(x, x_keep, p):

  output = torch.empty_like(x)
  assert x.is_contiguous()
  n_elem = output.numel()

  grid = lambda meta: (triton.cdiv(n_elem, meta['block_size']), )
  _dropout_kernel[grid](x, x_keep, output, n_elem, p, block_size = 1024)

  return output


x = torch.randn(size = (5,), device = device)
p = 0.5
x_keep = (torch.rand(size = (5,), device = device) > p).to(torch.int32)
output = dropout(x, x_keep, p)
print(x)
print(output)
