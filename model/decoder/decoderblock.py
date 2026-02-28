from torch import nn
from model.transformer.attention import MultiHeadAttention
from model.transformer.feedforward import FeedForwardBlock
from model.common.residual import ResidualConnection


class DecoderBlock(nn.Module):
    # attention:        masked self-attention — decoder tokens attend to each other
    #                   uses tgt_mask to prevent looking at future positions
    # cross_attention:  cross-attention — decoder queries the encoder's output
    #                   q comes from decoder, k and v come from encoder
    # feed_forward:     same structure as encoder's feedforward block
    # dropout:          passed to all three ResidualConnection instances
    def __init__(self, attention: MultiHeadAttention, cross_attention: MultiHeadAttention,
                 feed_forward: FeedForwardBlock, dropout):
        super().__init__()
        self.attention       = attention
        self.cross_attention = cross_attention
        self.feed_forward    = feed_forward
        # three residual connections — one per sublayer
        self.residual = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x:          (batch_size, tgt_seq_len, d_model) — decoder input
        # enc_output: (batch_size, src_seq_len, d_model) — frozen encoder output
        # src_mask:   prevents attending to <pad> tokens in the encoder output
        # tgt_mask:   causal mask — prevents decoder from seeing future tokens

        # step 1 — masked self-attention
        # q, k, v all come from x — decoder tokens attend to each other
        # tgt_mask enforces causality: token i can only see tokens 0..i
        x = self.residual[0](x, lambda x: self.attention(x, x, x, tgt_mask))

        # step 2 — cross-attention
        # q comes from decoder (x), k and v come from encoder output
        # this is where the decoder learns to focus on relevant source tokens
        # src_mask prevents attending to padding in the source sequence
        x = self.residual[1](x, lambda x: self.cross_attention(x, enc_output, enc_output, src_mask))

        # step 3 — feedforward
        # each token processed independently, same as in the encoder
        x = self.residual[2](x, self.feed_forward)

        # out: (batch_size, tgt_seq_len, d_model)
        return x