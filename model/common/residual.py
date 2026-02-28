import torch.nn as nn
from normalization import LayerNorm

class ResidualConnection(nn.Module):
    # dropout: applied to sublayer output before adding the residual
    #          prevents the sublayer from producing overly large updates
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        # x:        (batch_size, seq_len, d_model)
        # sublayer: a callable — either MultiHeadAttention or FeedForwardBlock
        #           passed as a function so this class stays generic and reusable
        #           across all encoder and decoder blocks

        # Pre-Norm pattern: normalize first, then apply sublayer, then add residual
        # the residual connection (+ x) solves the vanishing gradient problem —
        # gradients can flow directly back through the addition without passing
        # through the sublayer, keeping signal strong in deep networks
        x = x + self.dropout(sublayer(self.norm(x)))

        # out: (batch_size, seq_len, d_model) — shape unchanged
        return x