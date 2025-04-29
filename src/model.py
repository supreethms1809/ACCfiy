import torch
from torch import nn

class encoderModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(encoderModel, self).__init__()
        self.embeddings = nn.Embedding(input_dim, input_dim)
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=24)
        self.fc = nn.Linear(input_dim, output_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.fc(self.encoder(self.embeddings(x))))

class decoderModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(decoderModel, self).__init__()
        self.embeddings = nn.Embedding(input_dim, input_dim)
        self.decoderlayer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=8, batch_first=False)
        self.decoder = nn.TransformerDecoder(self.decoderlayer, num_layers=24)
        self.fc = nn.Linear(input_dim, output_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.fc(self.decoder(self.embeddings(x))))
    