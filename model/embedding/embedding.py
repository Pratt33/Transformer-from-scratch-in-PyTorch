import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    # vocab_size: total number of unique tokens in the vocabulary (words, punctuation,
    # special tokens like <bos>, <eos>, <pad>, and subword pieces)
    # d_model: the dimensionality of the embedding space (512 in our architecture)
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # nn.Embedding is a trainable lookup table of shape (vocab_size, d_model)
        # it maps each integer token ID to a dense vector of size d_model
        # this weight matrix is learned via backpropagation during training
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):
        # tokens: (batch_size, seq_len) — integer token IDs
        # raw embedding vectors are initialized small, while positional encoding
        # values are bounded between -1 and 1. scaling by sqrt(d_model) brings
        # the embeddings up to a comparable magnitude so neither dominates when added
        out = self.embedding(tokens) * math.sqrt(self.d_model)
        # out: (batch_size, seq_len, d_model)
        return out