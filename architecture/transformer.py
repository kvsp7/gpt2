import torch
from torch import nn

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            torch.nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        ) 

    def forward(self, x):
        return self.layers(x)

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdims=True,  unbiased=False)
        norm_x = (x - mean) / torch.sqrt(self.eps + var)
        bnorm_x = self.scale * norm_x + self.shift

        return bnorm_x

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

        self.dropout = torch.nn.Dropout(dropout)

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
        attention_scores.masked_fill_(mask_bool, -torch.inf)

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
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"]
        )

        self.ff = FeedForward(config=config)

        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])

        self.drop_shortcut = torch.nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


torch.manual_seed(123)

x = torch.rand(2, 4, 768)
block = TransformerBlock(config=GPT_CONFIG_124M)
o = block(x)

print(x.shape)
print(o.shape) 