import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    # d_model:    must match the decoder's output dimension (512)
    # vocab_size: total number of tokens in the target vocabulary
    #             output size must match so each position gets a score for every token
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # linear projection from d_model space into vocabulary space
        # maps each token's 512-dim representation to a score for every possible token
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, tgt_seq_len, d_model) — decoder output

        # step 1 — project to vocabulary dimension
        # (batch_size, tgt_seq_len, d_model) → (batch_size, tgt_seq_len, vocab_size)
        # each token position now has one raw score (logit) per vocabulary token

        # step 2 — log_softmax converts logits to log probabilities
        # dim=-1 means softmax is computed across vocab_size dimension independently
        # for each token position — the scores at each position sum to 1 in probability space
        # log_softmax is numerically more stable than log(softmax(x)) computed separately
        return torch.log_softmax(self.projection(x), dim=-1)

        # out: (batch_size, tgt_seq_len, vocab_size)
        # each position holds a log probability distribution over the entire vocabulary
        # the token with the highest value is the model's most likely prediction