import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length,
                  num_heads, dropout, qkv_bias=False):
        super().__init__()

        assert(d_out%num_heads == 0 ) , "d_out must be divisble by num_heads"

        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.num_heads = num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = torch.dropout(dropout)

        self.out_projection = nn.Linear(d_out, d_out)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self,x):
        batch_size, num_tokens, d_in = x.shape
        queries = self.W_query(x) 
        keys    = self.W_key(x)
        values  = self.W_value(x)

        keys    = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys    = keys.transpose(1,2)
        values  = values.transpose(1,2)
        queries = queries.transpose(1,2)

        attention_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.maksed_fill_(mask_bool, -torch.inf)

        dk = keys.shape[-1]
        attention_weights = torch.softmax(
            attention_scores / dk**0.5 , dim=-1 
        )
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1,2)

        context_vector = context_vector.contiguous().view(
            batch_size, num_tokens, self.d_out
        )
        context_vector = self.out_projection(context_vector)

        return context_vector

from tokenizer.Dataset_V1 import Data

batch = Data()

print(batch)
# batch_size, num_tokens, d_in = None
# attention = MultiHeadAttention(
#     d_in=d_in, d_out=d_in, context_length=num_tokens,
#     num_heads=12, dropout=0.5, qkv_bias=False
# )

