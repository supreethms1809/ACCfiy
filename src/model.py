# Custom Decoder -> Encoder -> Decoder with weight sharing in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
import os
import math
from tqdm import tqdm
import wandb

# Initialize wandb
wandb.init(project="decoder-encoder-decoder-prototype", name="run-1")

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.d_k = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = attn @ v
        context = context.transpose(1, 2).reshape(B, T, D)
        return self.out(context)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_hidden):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.ff = FeedForward(dim, ff_hidden)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, ff_hidden):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ff_hidden) for _ in range(depth)
        ])
        self.pos_enc = PositionalEncoding(dim)

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, mask)
        return x


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, ff_hidden):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ff_hidden) for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)


class DecoderEncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, ff_hidden):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.decoder = Decoder(dim, depth, heads, ff_hidden)
        self.encoder = Encoder(dim, depth=8, heads=heads, ff_hidden=ff_hidden)
        self.output_proj = nn.Linear(dim, vocab_size)

    def _causal_mask(self, x):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        mask = self._causal_mask(x)
        hidden1 = self.decoder(x, mask=mask)
        z = self.encoder(hidden1).unsqueeze(1).expand(-1, hidden1.size(1), -1)
        hidden_combined = hidden1 + z
        hidden2 = self.decoder(hidden_combined, mask=mask)
        return self.output_proj(hidden2)

    def generate(self, input_ids, max_new_tokens=50):
        for _ in range(max_new_tokens):
            output = self.forward(input_ids)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token.transpose(0, 1)], dim=1)
        return input_ids


class NextTokenDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
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


# --- Train Script ---
if __name__ == "__main__":
    accelerator = Accelerator(mixed_precision="fp16")

    raw_dataset = load_dataset("xsum", split="train[:1000]")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-1.3B-base", trust_remote_code=True)
    train_data = raw_dataset.select(range(900))
val_data = raw_dataset.select(range(900, 1000))
dataset = NextTokenDataset(train_data, tokenizer)
val_dataset = NextTokenDataset(val_data, tokenizer)

    def collate(batch):
        input_seqs = [x[0] for x in batch]
        target_seqs = [x[1] for x in batch]
        input_padded = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
        target_padded = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
        return input_padded, target_padded

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

    model = DecoderEncoderDecoderModel(len(tokenizer), dim=512, depth=28, heads=8, ff_hidden=2048)
    start_epoch = 0
    if os.path.exists("checkpoint.pt"):
        print("Loading checkpoint.pt to resume training...")
        checkpoint = torch.load("checkpoint.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=0)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}.")
    else:
        print("Training from scratch.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(loader) * 100

    # Load best model checkpoint if available
    if os.path.exists("best_model.pt"):
        print("Loading best_model.pt to resume training...")
        model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
        print("Resumed model from checkpoint.")
    else:
        print("Training from scratch.")
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    best_loss = float("inf")
    losses_per_epoch = []

    for epoch in range(start_epoch, 100):
        model.train()
        total_loss = 0
        total_tokens = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for step, (input_ids, target_ids) in enumerate(progress):
            input_ids = input_ids.to(accelerator.device)
            target_ids = target_ids.to(accelerator.device)
            output = model(input_ids)
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            loss = loss / 2  # Gradient accumulation step (e.g., every 2 steps)
            accelerator.backward(loss)

            if (step + 1) % 2 == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            scheduler.step()
            total_loss += loss.item()
            total_tokens += target_ids.numel()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        perplexity = math.exp(total_loss / total_tokens)
        losses_per_epoch.append(avg_loss)
                # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_tokens = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(accelerator.device)
                target_ids = target_ids.to(accelerator.device)
                output = model(input_ids)
                loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
                val_loss += loss.item()
                val_tokens += target_ids.numel()

        val_avg_loss = val_loss / len(val_loader)
        val_perplexity = math.exp(val_loss / val_tokens)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, PPL: {perplexity:.2f} | Val Loss: {val_avg_loss:.4f}, Val PPL: {val_perplexity:.2f}")

        wandb.log({"train_loss": avg_loss, "train_perplexity": perplexity,
                   "val_loss": val_avg_loss, "val_perplexity": val_perplexity, "epoch": epoch + 1})

        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch
            }
            accelerator.save(checkpoint, "checkpoint.pt")
            print("Checkpoint saved.")

        # Log to wandb
        wandb.log({"loss": avg_loss, "perplexity": perplexity, "epoch": epoch + 1})

        if avg_loss < best_loss:
            best_loss = avg_loss
            accelerator.save(model.state_dict(), "best_model.pt")
            print("Model saved.")

    model.eval()
    prompt = "The economy in Europe"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(accelerator.device)
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_new_tokens=50)
        print("Generated text:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    for i, loss in enumerate(losses_per_epoch):
        wandb.log({"epoch_loss": loss, "epoch": i + 1})

    wandb.finish()
