import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import tiktoken
from itertools import chain

class TokenizedDataset(IterableDataset):
    def __init__(self, dataset_name, split, max_seq_len, batch_size, tokenizer_model="gpt-4"):
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_len = max_seq_len
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        self.batch_size = batch_size
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def __iter__(self):
        # Create an iterator from the dataset
        iterator = iter(self.dataset)
        
        # Buffer to hold tokens
        buffer = []
        
        for item in iterator:
            text = item.get('text', '')
            if not text:
                continue
                
            tokens = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
            tokens.append(self.tokenizer.eot_token) # Add EOS token
            buffer.extend(tokens)
            
            # Yield chunks of max_seq_len + 1 (input + target)
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1:]
                
                # Convert to tensor
                yield torch.tensor(chunk, dtype=torch.long)
                
    def get_vocab_size(self):
        return self.tokenizer.n_vocab

def collate_fn(batch):
    # Stack the list of tensors
    # Batch shape: (batch_size, seq_len + 1)
    return torch.stack(batch)

def get_dataloader(dataset_name, split, max_seq_len, batch_size, num_workers=0):
    dataset = TokenizedDataset(dataset_name, split, max_seq_len, batch_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader, dataset.get_vocab_size()
