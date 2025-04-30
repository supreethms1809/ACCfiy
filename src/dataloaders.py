from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch

class NextTokenDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['document'][:self.max_len]
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")['input_ids'].squeeze(0)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return input_ids, target_ids

class DataLoaders():
    def __init__(self, name, batch_size, tokenizer, max_length, size=None, split=0.9):
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

    def load_data(self):
        raw_dataset = load_dataset(self.name, split='train')
        train_size = int(len(raw_dataset) * self.split)
        val_size = len(raw_dataset) - train_size
        train_dataset = raw_dataset.shuffle().select(range(train_size))
        val_dataset = raw_dataset.shuffle().select(range(train_size, train_size + val_size))
        return train_dataset, val_dataset

    def create_dataloaders(self):
        train_dataset, val_dataset = self.load_data()
        train_dataset = NextTokenDataset(train_dataset, self.tokenizer, self.max_length)
        val_dataset = NextTokenDataset(val_dataset, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)

        return train_loader, val_loader