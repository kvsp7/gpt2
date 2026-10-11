from architecture.transformer import TransformerBlock,LayerNorm
import torch
from config import GPT_CONFIG_124M


class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_embed = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = torch.nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_embed = torch.nn.Dropout(config["drop_rate"])

        self.transformer_blocks = torch.nn.Sequential(
            *[TransformerBlock(config=config) for _ in range(config["n_layers"])]
        )

        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = torch.nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeddings = self.tok_embed(in_idx)
        pos_embeddings = self.pos_embed(
            torch.arange(seq_len, device=in_idx.device)
        ) 

        x = tok_embeddings + pos_embeddings
        x = self.drop_embed(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


# torch.manual_seed(7)

# model = GPTModel(config=GPT_CONFIG_124M)

# import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")

# b = [tokenizer.encode("hey guys hello bro"), tokenizer.encode("nice to meet you")]
# b = torch.tensor(b)
# print(b.shape)
# l = model(b)

# print(l.shape)

# total_params = sum(p.numel() for p in model.parameters())
# total_gpt2_params = total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"{total_gpt2_params:,}")


# total_size_bytes = total_params * 4
# total_size_mb = total_size_bytes / (1024 * 1024)
# print(f"total mb is {total_size_mb:.4f}")