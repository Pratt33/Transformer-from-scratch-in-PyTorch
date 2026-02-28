import torch.nn as nn
from model.common.normalization import LayerNorm

class Encoder(nn.Module):
    # layer: nn.ModuleList of N EncoderBlock instances
    #        each block has its own independent weights
    #        N is a hyperparameter — 6 in the original paper
    def __init__(self, layer: nn.ModuleList):
        super().__init__()
        self.layer = layer
        # final norm applied after all blocks — stabilizes the output
        # before it's passed to the decoder or output projection
        self.norm = LayerNorm()

    def forward(self, x, src_mask):
        # x:        (batch_size, seq_len, d_model)
        # src_mask: passed through to every block unchanged —
        #           each block needs it for the self-attention step
        for layer in self.layer:
            x = layer(x, src_mask)

        # normalize the final output of the encoder stack
        # out: (batch_size, seq_len, d_model) — shape unchanged
        return self.norm(x)