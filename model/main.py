import torch.nn as nn
from encoder.encoder import Encoder
from encoder.encoderblock import EncoderBlock
from decoder.decoder import Decoder
from decoder.decoderblock import DecoderBlock
from embedding.embedding import TokenEmbedding
from embedding.encoding import PositionalEncoding
from transformer.attention import MultiHeadAttention
from transformer.feedforward import FeedForwardBlock
from decoder.projection import ProjectionLayer
from transformer.transformer import Transformer
from config import device


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len,
                      d_model, N, h, dropout, d_ff):
    # --- Embeddings ---
    # separate embedding tables for source and target — they may have different vocabularies
    # e.g. English → French translation has two distinct vocab sets
    src_embedding = TokenEmbedding(src_vocab_size, d_model)
    tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

    # separate positional encodings for source and target sequences
    # src and tgt may have different max lengths
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # --- Encoder ---
    # each block gets its own fresh MultiHeadAttention and FeedForwardBlock instances
    # this is critical — shared weights across blocks would mean all blocks learn
    # identical transformations, defeating the purpose of stacking N layers
    encoder_blocks = []
    for _ in range(N):
        encoder_attention    = MultiHeadAttention(d_model, h, dropout)
        encoder_feedforward  = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block        = EncoderBlock(encoder_attention, encoder_feedforward, dropout)
        encoder_blocks.append(encoder_block)

    # --- Decoder ---
    # each decoder block needs THREE independent component instances:
    # self-attention, cross-attention, and feedforward — all with separate weights
    decoder_blocks = []
    for _ in range(N):
        decoder_attention       = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_feedforward     = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block           = DecoderBlock(decoder_attention, decoder_cross_attention,
                                               decoder_feedforward, dropout)
        decoder_blocks.append(decoder_block)

    # --- Assemble ---
    encoder   = Encoder(nn.ModuleList(encoder_blocks))
    decoder   = Decoder(nn.ModuleList(decoder_blocks))
    projection = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding,
                              src_pos, tgt_pos, projection)

    # --- Weight Initialization ---
    # Xavier uniform initialization sets weights to values drawn from a uniform
    # distribution scaled by the layer's fan-in and fan-out. This keeps the
    # variance of activations consistent across layers at the start of training,
    # preventing vanishing or exploding gradients from the very first forward pass.
    # dim > 1 skips biases and 1D parameters — only weight matrices are initialized
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # move entire model and all its parameters to the correct device in one call
    return transformer.to(device)