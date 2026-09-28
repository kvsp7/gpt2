import torch

torch.manual_seed(1)

class CasualAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.dropout = torch.nn.Dropout(dropout)
        self.Wq = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_int = x.shape
        queries = self.Wq(x)
        keys    = self.Wk(x)
        values  = self.Wv(x)
        attention_scores = queries @ keys.transpose(1,2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        dk = keys.shape[-1]
        attention_weights = torch.softmax(
            attention_scores / dk**0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ values
        
        return context_vec

from tokenizer.Dataset_V1 import Data

data = Data()

d_in = data.shape[-1]
d_out = 4
cl = data.shape[1]
drop=0.5

# s = CasualAttention(d_in=d_in, d_out=d_out,context_length=cl,dropout=d)
# f = s.forward(e)
'''
    Here a stack of instances is being created.
'''
heads = torch.nn.ModuleList(
    [CasualAttention(
        d_in=d_in, d_out=d_out, context_length=cl, dropout=drop
        )
     for _ in range(2)]
)

context = []
for head in heads():
    context_vec = head(data)
    context.append(context_vec)
print(heads.shape)
