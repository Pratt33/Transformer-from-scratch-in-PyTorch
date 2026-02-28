# decoder.py
import torch.nn as nn
from model.common.normalization import LayerNorm


class Decoder(nn.Module):
    # layer: nn.ModuleList of N DecoderBlock instances
    #        each block has its own independent weights for all three sublayers
    def __init__(self, layer: nn.ModuleList):
        super().__init__()
        self.layer = layer
        # final norm applied after all blocks, mirrors the encoder structure
        self.norm = LayerNorm()

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x:          (batch_size, tgt_seq_len, d_model)
        # enc_output: (batch_size, src_seq_len, d_model) — same tensor passed
        #             to every decoder block unchanged, each block can attend to it
        # src_mask and tgt_mask passed through to every block unchanged
        for layer in self.layer:
            x = layer(x, enc_output, src_mask, tgt_mask)

        # out: (batch_size, tgt_seq_len, d_model)
        return self.norm(x)