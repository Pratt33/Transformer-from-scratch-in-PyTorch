import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    # d_model: input and output dimension — must stay consistent across all blocks
    # d_ff:    inner dimension, typically 4 * d_model (2048 in our architecture)
    #          this expansion gives the model more capacity to learn complex transformations
    #          before projecting back down to d_model
    # dropout: applied after activation to prevent overfitting
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        # first projection: expands from d_model -> d_ff
        # nn.Linear internally holds weight matrix (d_model, d_ff) and bias (d_ff,)
        self.linear1 = nn.Linear(d_model, d_ff)   # W1 and b1

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # second projection: compresses back from d_ff -> d_model
        # output dimension matches input so blocks can be stacked
        self.linear2 = nn.Linear(d_ff, d_model)   # W2 and b2

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # nn.Linear applies the transformation to the last dimension only,
        # leaving batch and seq_len dimensions completely untouched

        x = self.linear1(x)   # (batch_size, seq_len, d_ff)
        x = self.gelu(x)      # non-linearity — without this, two linear layers
                               # would collapse into one, adding no extra capacity
        x = self.dropout(x)   # randomly zero out activations during training only
        x = self.linear2(x)   # (batch_size, seq_len, d_model)

        # out: (batch_size, seq_len, d_model) — shape restored to match input
        return x