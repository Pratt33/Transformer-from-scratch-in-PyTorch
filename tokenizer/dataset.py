import torch
from torch.utils.data import Dataset


def causal_mask(size):
    # creates a lower triangular matrix of ones — position i can attend
    # to positions 0..i but not i+1..size (future positions)
    # shape: (1, size, size) — the 1 allows broadcasting over batch dimension
    mask = torch.tril(torch.ones(1, size, size)).type(torch.bool)
    # return True where attention IS allowed (lower triangle including diagonal)
    return mask == 1


class BilingualDataset(Dataset):
    # ds:            one split only — train_ds, val_ds, or test_ds
    # tokenizer_src: trained BPE tokenizer for source language
    # tokenizer_tgt: trained BPE tokenizer for target language
    # src_lang:      language code string e.g. 'en'
    # tgt_lang:      language code string e.g. 'fr'
    # seq_len:       fixed sequence length — all sequences padded or truncated to this
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds            = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang      = src_lang
        self.tgt_lang      = tgt_lang
        self.seq_len       = seq_len

        # store special token IDs as scalar tensors
        # using tokenizer_tgt for all because decoder_input and label
        # both live in the target vocabulary space
        self.sos_token = torch.LongTensor([tokenizer_tgt.token_to_id('[SOS]')])
        self.eos_token = torch.LongTensor([tokenizer_tgt.token_to_id('[EOS]')])
        self.pad_token = torch.LongTensor([tokenizer_tgt.token_to_id('[PAD]')])

    def __len__(self):
        # tells DataLoader how many samples exist in this split
        return len(self.ds)

    def __getitem__(self, idx):
        # fetch one src-tgt sentence pair by index
        src_tgt_pair = self.ds[idx]
        src_text     = src_tgt_pair['translation'][self.src_lang]
        tgt_text     = src_tgt_pair['translation'][self.tgt_lang]

        # tokenize both sentences into lists of integer IDs
        # encode() returns an Encoding object — .ids gives the integer list
        enc_input_ids = self.tokenizer_src.encode(src_text).ids
        dec_input_ids = self.tokenizer_tgt.encode(tgt_text).ids

        # truncate to leave room for special tokens
        # encoder needs [SOS] and [EOS] → reserve 2 positions
        # decoder input needs [SOS] only → reserve 1 position
        # label needs [EOS] only → reserve 1 position
        enc_input_ids = enc_input_ids[:self.seq_len - 2]
        dec_input_ids = dec_input_ids[:self.seq_len - 1]

        # calculate how many [PAD] tokens are needed to reach seq_len
        enc_num_pad = self.seq_len - len(enc_input_ids) - 2
        dec_num_pad = self.seq_len - len(dec_input_ids) - 1

        if enc_num_pad < 0 or dec_num_pad < 0:
            raise ValueError(f"Sequence too long at index {idx}")

        # encoder input: [SOS] + tokens + [EOS] + [PAD]...
        # the encoder sees the full source sentence with boundary markers
        encoder_input = torch.cat([
            self.sos_token,
            torch.LongTensor(enc_input_ids),
            self.eos_token,
            torch.LongTensor([self.pad_token.item()] * enc_num_pad)
        ])

        # decoder input: [SOS] + tokens + [PAD]...
        # [SOS] prompts the decoder to start generating
        # notably NO [EOS] here — the decoder hasn't finished yet
        decoder_input = torch.cat([
            self.sos_token,
            torch.LongTensor(dec_input_ids),
            torch.LongTensor([self.pad_token.item()] * dec_num_pad)
        ])

        # label: tokens + [EOS] + [PAD]...
        # what the decoder SHOULD have predicted at each position
        # shifted one step ahead of decoder_input — this is teacher forcing
        # [EOS] tells the model when the sequence is complete
        label = torch.cat([
            torch.LongTensor(dec_input_ids),
            self.eos_token,
            torch.LongTensor([self.pad_token.item()] * dec_num_pad)
        ])

        # verify all sequences are exactly seq_len — catches any construction bugs
        assert encoder_input.size(0) == decoder_input.size(0) == label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,   # (seq_len,)
            'decoder_input': decoder_input,   # (seq_len,)
            'label':         label,           # (seq_len,)

            # encoder mask: 1 where token is real, 0 where [PAD]
            # shape: (1, 1, seq_len) — broadcasts over batch and head dimensions
            'encoder_mask': (encoder_input != self.pad_token.item())
                            .unsqueeze(0).unsqueeze(0).bool(),

            # decoder mask: combines padding mask AND causal mask
            # padding mask shape: (1, 1, seq_len)
            # causal mask shape:  (1, seq_len, seq_len)
            # & broadcasts padding mask across all rows of causal mask
            # result: (1, seq_len, seq_len) — each position only attends to
            # non-padding positions that came before it
            'decoder_mask': (decoder_input != self.pad_token.item())
                            .unsqueeze(0).unsqueeze(0).bool()
                            & causal_mask(decoder_input.size(0)),

            # raw text kept for inspection and BLEU evaluation later
            'src_text': src_text,
            'tgt_text': tgt_text
        }