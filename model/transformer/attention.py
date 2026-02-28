import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    # d_model:    full embedding dimension (512) — split equally across heads
    # num_heads:  number of parallel attention heads (8)
    #             each head attends to different aspects of the sequence independently
    # dropout:    applied to attention weights to prevent over-attending to specific tokens
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.att_score = None  # stored for inspection/visualization if needed
        self.d_model   = d_model
        self.num_heads = num_heads
        # d_k: dimension each head operates in — heads work in lower-dimensional subspaces
        # and their outputs are concatenated back to d_model
        self.d_k = d_model // num_heads  # 512 // 8 = 64

        # each projection learns a different linear transformation of the input
        self.wq = nn.Linear(d_model, d_model)  # what each token is looking for
        self.wk = nn.Linear(d_model, d_model)  # what each token advertises it contains
        self.wv = nn.Linear(d_model, d_model)  # what each token actually gives away

        # projects concatenated head outputs back to d_model
        # learns how to blend information from all heads into one unified representation
        self.wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        # query, key, value: (batch_size, num_heads, seq_len, d_k)
        d_k = query.size(-1)

        # dot product measures alignment between query and key vectors
        # scaling by sqrt(d_k) prevents dot products from growing too large,
        # which would push softmax into regions with near-zero gradients
        # (batch_size, num_heads, seq_len, seq_len)
        att_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # set masked positions to large negative value so softmax gives them ~0 weight
            att_score = att_score.masked_fill(mask == 0, -1e9)

        # softmax across last dim — each query position gets a probability
        # distribution over all key positions
        att_score = F.softmax(att_score, dim=-1)

        if dropout is not None:
            att_score = dropout(att_score)

        # weighted sum of value vectors — output shape: (batch_size, num_heads, seq_len, d_k)
        return (att_score @ value), att_score

    def forward(self, q, k, v, mask):
        B, S, D = q.shape
        # q, k, v: (batch_size, seq_len, d_model)

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # split d_model into num_heads heads of size d_k each
        # view:      (batch_size, seq_len, num_heads, d_k)
        # transpose: (batch_size, num_heads, seq_len, d_k)
        # after transpose each head has its own full sequence — this is what allows
        # each head to independently compute attention across all token positions
        q = q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        # run attention independently for all heads simultaneously (parallelized)
        x, self.att_score = MultiHeadAttention.attention(q, k, v, mask, self.dropout)

        # recombine heads:
        # transpose: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        # view:      (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        # contiguous() is required before view() because transpose creates a non-contiguous
        # tensor in memory — view() needs contiguous memory to reshape safely
        x = x.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # final linear projection: (batch_size, seq_len, d_model)
        return self.wo(x)