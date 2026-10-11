import torch
import tiktoken
from GPT_model import GPTModel
from config import GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_clip = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_clip)
        logits = logits[:, -1,:] #like last token of a batch 
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

start_context = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print(encoded_tensor.shape, encoded_tensor)

model.eval()

out = generate_text_simple(
    model=GPTModel(GPT_CONFIG_124M),
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

# print(out)

out_list = out.squeeze(0).tolist()

print(tokenizer.decode(out_list))