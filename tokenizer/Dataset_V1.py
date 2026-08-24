import torch
import tiktoken
from torch.utils.data import Dataset,DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return ( torch.tensor(self.input_ids[idx], dtype=torch.long),
                 torch.tensor(self.target_ids[idx], dtype=torch.long))
    



def create_dataloaderV1(text, tokenizer, batch_size=4, max_length = 256,
                        shuffle=True, drop_last=True, num_workers=0, stride=128):
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset = dataset,
        shuffle = shuffle,
        batch_size = batch_size,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader,max_length

def Data():
    with open(r"The-verdict.txt","r",encoding='utf-8') as file:
        text = file.read()

    tokenizer = tiktoken.get_encoding('gpt2')

    dataloader, max_length = create_dataloaderV1(text=text, tokenizer=tokenizer, batch_size=4, max_length=256,
                                    stride=128, shuffle=False)

    vocab_size = tokenizer.n_vocab
    output_dims = 256

    embedding_layer = torch.nn.Embedding(vocab_size, output_dims)
    pos_embedding_layer = torch.nn.Embedding(max_length, output_dims)

    data = []
    for x,y in dataloader:
        data.append((x, y))

    return data

