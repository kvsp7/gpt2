import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb   = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb  = nn.Dropout(config["drop_rate"])
        self.trf_block = nn.Sequential(
            *[DummyTransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(config["emb_dim"])
        self.out_head   = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')

batch = ["Every effort moves you",
         "Every day holds a"]
batch = [torch.tensor(tokenizer.encode(sent)) for sent in batch]
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)

model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)

print(logits.shape)
print(logits)