import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2048):
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
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        #     # p.requires_grad = True  # Uncomment to train encoder

    def _causal_mask(self, x):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        mask = self._causal_mask(x)

        hidden1 = self.decoder(x, mask=mask)
        with torch.no_grad():  # Freeze encoder
            z = self.encoder(hidden1).unsqueeze(1).expand(-1, hidden1.size(1), -1)
        hidden_combined = hidden1 + z
        hidden2 = self.decoder(hidden_combined, mask=mask)

        logits1 = self.output_proj(hidden1)
        logits2 = self.output_proj(hidden2)
        return logits1, logits2

    # def forward(self, input_ids):
    #     x = self.token_embedding(input_ids)
    #     mask = self._causal_mask(x)
    #     hidden1 = self.decoder(x, mask=mask)
    #     z = self.encoder(hidden1).unsqueeze(1).expand(-1, hidden1.size(1), -1)
    #     hidden_combined = hidden1 + z
    #     hidden2 = self.decoder(hidden_combined, mask=mask)
    #     return self.output_proj(hidden2)

    def generate(self, input_ids, max_new_tokens=50):
        for _ in range(max_new_tokens):
            x = self.token_embedding(input_ids)
            mask = self._causal_mask(x)
            hidden1 = self.decoder(x, mask=mask)
            z = self.encoder(hidden1).unsqueeze(1).expand(-1, hidden1.size(1), -1)
            hidden_combined = hidden1 + z
            hidden2 = self.decoder(hidden_combined, mask=mask)
            logits = self.output_proj(hidden2)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token.transpose(0, 1)], dim=1)
        return input_ids

