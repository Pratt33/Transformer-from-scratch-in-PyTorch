import torch
import torch.nn as nn
from model.encoder.encoder import Encoder
from model.decoder.decoder import Decoder
from model.embedding.embedding import TokenEmbedding
from model.embedding.encoding import PositionalEncoding
from model.decoder.projection import ProjectionLayer


class Transformer(nn.Module):
    # all components are constructed externally and injected here —
    # this keeps the Transformer class clean and makes each component
    # independently testable before being wired together
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: TokenEmbedding, tgt_embed: TokenEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection: ProjectionLayer):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.src_embed  = src_embed
        self.tgt_embed  = tgt_embed
        self.src_pos    = src_pos
        self.tgt_pos    = tgt_pos
        self.projection = projection

    def encode(self, src, src_mask):
        # src: (batch_size, src_seq_len) — source token IDs
        src = self.src_embed(src)       # (batch_size, src_seq_len, d_model)
        src = self.src_pos(src)         # (batch_size, src_seq_len, d_model)
        return self.encoder(src, src_mask)  # (batch_size, src_seq_len, d_model)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        # tgt: (batch_size, tgt_seq_len) — target token IDs
        tgt = self.tgt_embed(tgt)       # (batch_size, tgt_seq_len, d_model)
        tgt = self.tgt_pos(tgt)         # (batch_size, tgt_seq_len, d_model)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)  # (batch_size, tgt_seq_len, d_model)

    def project(self, x):
        # x: (batch_size, tgt_seq_len, d_model)
        return self.projection(x)       # (batch_size, tgt_seq_len, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # full forward pass wiring encode → decode → project
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.project(dec_output)