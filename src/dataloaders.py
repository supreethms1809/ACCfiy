from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch

class NextTokenDataset(Dataset):
    def __init__(self, data, tokenizer, col_name, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.col_name = col_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.col_name == 'document':
            text = self.data[idx]['document'][:self.max_len]
        elif self.col_name == 'text':
            text = self.data[idx]['text'][:self.max_len]
        else:
            raise ValueError(f"Column name {self.col_name} is not supported. Use 'document' or 'text'.")
        #text = self.data[idx]['document'][:self.max_len]
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")['input_ids'].squeeze(0)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return input_ids, target_ids

class DataLoaders():
    def __init__(self, name, batch_size, tokenizer, max_length, size=None, split=0.9, num_workers=4):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.name = name
        self.size = size
        self.split = split

    def collate(self, batch):
        input_seqs = [x[0] for x in batch]
        target_seqs = [x[1] for x in batch]
        input_padded = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        target_padded = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return input_padded, target_padded

    def load_data_xsum(self):
        raw_dataset = load_dataset(self.name, split='train')
        train_size = int(len(raw_dataset) * self.split)
        val_size = len(raw_dataset) - train_size
        train_dataset = raw_dataset.shuffle().select(range(train_size))
        val_dataset = raw_dataset.shuffle().select(range(train_size, train_size + val_size))
        return train_dataset, val_dataset
    
    def load_dataset_c4(self):
        raw_dataset = load_dataset(self.name, 'en', split='train')
        # if self.size != None:
        #     raw_dataset = raw_dataset.shuffle().select(range(self.size))
        train_size = int(len(raw_dataset) * self.split)
        val_size = len(raw_dataset) - train_size
        train_dataset = raw_dataset.shuffle().select(range(train_size))
        val_dataset = raw_dataset.shuffle().select(range(train_size, train_size + val_size))
        return train_dataset, val_dataset

    def create_dataloaders(self):
        if self.name == "xsum":
            train_dataset, val_dataset = self.load_data_xsum()
            col_name = 'document'
        elif self.name == "allenai/c4":
            train_dataset, val_dataset = self.load_dataset_c4()
            col_name = 'text'
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

        train_dataset = NextTokenDataset(train_dataset, self.tokenizer, col_name, self.max_length)
        val_dataset = NextTokenDataset(val_dataset, self.tokenizer, col_name, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate)

        return train_loader, val_loader