import torch
context_length=256
a = torch.tensor(torch.arange(36, dtype=torch.float32).reshape(6,6))
mask = torch.triu(torch.ones(context_length, context_length),diagonal=1)
masked = a.masked_fill(mask.bool(),-torch.inf)
print(a)
print(mask.bool())
a = torch.tensor([[1]],dtype=torch.long)
