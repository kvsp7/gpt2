import torch

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.t                    55   `anh(
            torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

'''
 ReLU vs GELU

 import matplotlib.pyplot as plt
g,r = GELU(), torch.nn.ReLU()
x = torch.linspace(-3, 3, 100)

yg, yr = g(x), r(x)

plt.figure(figsize=(8,3))

for i, (y, l) in enumerate(zip([yg,yr], ["GELU","ReLU"]), 1):
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.grid(True)
plt.show()

'''