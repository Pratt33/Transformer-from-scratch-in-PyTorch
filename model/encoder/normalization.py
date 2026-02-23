import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    # eps: small constant added to std to prevent division by zero
    # when all values in a vector are identical, std = 0 and we'd get NaN without it
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        # alpha and bias are learned parameters — the model learns how much
        # to scale and shift each normalized value during training.
        # without these, normalization would force every layer's output
        # into the same fixed distribution, removing the model's expressive power
        self.alpha = nn.Parameter(torch.ones(1))   # multiplicative scale
        self.bias  = nn.Parameter(torch.zeros(1))  # additive shift

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # normalize across the last dimension (d_model) independently for each token
        # this means each token's 512-dimensional vector is normalized on its own,
        # completely independent of other tokens in the sequence or batch
        mean = x.mean(-1, keepdim=True)  # (batch_size, seq_len, 1)
        std  = x.std (-1, keepdim=True)  # (batch_size, seq_len, 1)

        # normalize, then rescale and shift with learned parameters
        # keepdim=True above ensures broadcasting works correctly across d_model
        out = (x - mean) * self.alpha / (std + self.eps) + self.bias
        # out: (batch_size, seq_len, d_model) — shape unchanged
        return out