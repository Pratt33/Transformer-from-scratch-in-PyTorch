import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # d_model: must match TokenEmbedding's d_model so shapes align when added
    # seq_len: maximum sequence length the model can handle
    # dropout: applied after adding positional encoding, regularizes the input representation
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # pe: (seq_len, d_model) — one position vector per token position
        pe = torch.zeros(seq_len, d_model)

        # position: (seq_len, 1) — each row is a token position index
        position = torch.arange(0, seq_len).unsqueeze(1).float()

        # div_term: (d_model/2,) — decreasing frequencies for each dimension pair
        # high-frequency dimensions capture fine-grained local position differences
        # low-frequency dimensions capture coarse global position differences
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sin to even indices, cos to odd indices
        # each position gets a unique fingerprint across all d_model dimensions
        pe[:, 0::2] = torch.sin(position * div_term)  # (seq_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (seq_len, d_model/2)

        # register as buffer: saved with model but not trained via backpropagation
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:  (batch_size, seq_len, d_model) — comes from TokenEmbedding
        # pe: (seq_len, d_model) — slice to actual sequence length, broadcast over batch
        # detach() ensures no gradients flow through pe since it is not learned
        x = x + self.pe[:x.size(1), :].detach()
        # out: (batch_size, seq_len, d_model) — shape unchanged, position info added
        return self.dropout(x)