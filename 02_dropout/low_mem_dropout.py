mport triton
import torch
import triton.language as tl

device= triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _low_memory_dropout(x_ptr, op_ptr, seed, p, n_elem, block_size: tl.constexpr):

  pid = tl.program_id(axis = 0)
  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)

  mask = offsets < n_elem
  x = tl.load(x_ptr + offsets, mask = mask)
  random = tl.rand(seed, offsets)
  x_keep = random > p

  output = tl.where(x_keep, x / (1 - p), 0.0)
  tl.store(op_ptr + offsets, output, mask = mask)

def seeded_dropout(x, seed, p):

  output = torch.empty_like(x)
  assert x.is_contiguous()
  n_elem = output.numel()
  grid = lambda meta: ((triton.cdiv(n_elem, meta['block_size'])), )

  _low_memory_dropout[grid](x, output, seed, p, n_elem, block_size = 1024)
  return output


x = torch.randn(size=(10, ), device= device)
p = 0.5
output = seeded_dropout(x, seed = 123, p = p)
output2 = seeded_dropout(x, seed = 123, p = p)
output3 = seeded_dropout(x, seed = 512, p = p)

print(output)
print(output2)
print(output3)
