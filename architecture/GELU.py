import torch

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

import matplotlib.pyplot as plt
g,r = GELU(), torch.nn.ReLU()
x = torch.linspace(-3, 3, 100)

yg, yr = g(x), r(x)

plt.figure(figsize=(8,3))


plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.grid(True)
plt.show()