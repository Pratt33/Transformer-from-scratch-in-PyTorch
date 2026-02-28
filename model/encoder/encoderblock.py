import torch.nn as nn
from model.transformer.attention import MultiHeadAttention
from model.transformer.feedforward import FeedForwardBlock
from model.common.residual import ResidualConnection

class EncoderBlock(nn.Module):
    # attention:    pre-built MultiHeadAttention instance — passed in rather than
    #               constructed here so each encoder block can have its own independent weights
    # feed_forward: pre-built FeedForwardBlock instance — same reasoning
    # dropout:      passed to ResidualConnection to regularize sublayer outputs
    def __init__(self, attention: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout):
        super().__init__()
        self.attention    = attention
        self.feed_forward = feed_forward

        # two residual connections — one wrapping attention, one wrapping feed_forward
        # ModuleList ensures PyTorch tracks these as registered submodules
        # so their parameters are included in model.parameters()
        self.residual = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # x:        (batch_size, seq_len, d_model)
        # src_mask: prevents tokens from attending to <pad> positions

        # step 1 — self-attention with residual connection
        # lambda is needed because attention takes multiple arguments (q, k, v, mask)
        # but ResidualConnection expects a single-argument callable
        # q, k, v are all x here because in the encoder every token attends to every other token
        x = self.residual[0](x, lambda x: self.attention(x, x, x, src_mask))

        # step 2 — feed forward with residual connection
        # no lambda needed — feed_forward takes a single input x
        x = self.residual[1](x, self.feed_forward)

        # out: (batch_size, seq_len, d_model) — shape unchanged throughout
        return x