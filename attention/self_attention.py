import torch

class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.Wq = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        atten_scores = queries @ keys.T
        dk = keys.shape[-1]
        atten_weights = torch.softmax(
            atten_scores / dk**0.5, dim=-1
        )
        context_vec = atten_weights @ values
        return context_vec


