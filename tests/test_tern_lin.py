from src.quantize import TernaryLinear
import torch
import torch.nn as nn

orig = nn.Linear(4, 2)
tern = TernaryLinear.from_linear(orig)

x = torch.randn(1, 4)
out = tern(x)
loss = out.sum()
loss.backward()

print('output:', out)
print('weight grad exists:', tern.weight.grad is not None)
print('weight grad:', tern.weight.grad)