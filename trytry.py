import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad

x1 = torch.tensor([3., 4.], requires_grad=True)
x2 = torch.exp(x1)
print(x2)
x3 = torch.square(x2)
y = torch.sum(x3)


gd1 = torch.tensor([0., 0.])

print(x3)

z = torch_grad(x3, x1, grad_outputs=gd1)
print(z)


