import torch
from tokenizer.dataset import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # encode source once — output reused at every decoding step
    encoder_output = model.encode(source, source_mask)  # (1, src_seq_len, d_model)

    # initialize decoder input with [SOS]
    decoder_input = torch.full((1, 1), sos_idx, dtype=source.dtype, device=device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # causal mask size grows with each new token appended
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)

        out  = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        prob = model.project(out[:, -1])  # project last position only: (1, vocab_size)

        # greedy: always pick highest probability token
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([
            decoder_input,
            next_word.unsqueeze(0)   # (1, 1)
        ], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)  # (seq_len,)


def beam_search_decode(model, source, source_mask, tokenizer_tgt, max_len, device, beam_size=4):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    # each beam: (sequence_tensor, cumulative_score)
    beams     = [(torch.tensor([[sos_idx]], device=device), 0.0)]
    completed = []

    for _ in range(max_len):
        new_beams = []

        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                completed.append((seq, score))
                continue

            decoder_mask = causal_mask(seq.size(1)).to(device)
            out          = model.decode(seq, encoder_output, source_mask, decoder_mask)
            log_probs    = torch.log_softmax(model.project(out[:, -1]), dim=-1)

            topk_probs, topk_idx = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_token = topk_idx[0, i].unsqueeze(0).unsqueeze(0)  # (1, 1)
                new_seq    = torch.cat([seq, next_token], dim=1)

                # length-normalized score — prevents bias toward shorter sequences
                # each added token contributes a negative log prob, so raw cumulative
                # score always decreases — longer sequences are unfairly penalized
                # dividing by sequence length puts all beams on equal footing
                #new_score = (score * (seq.size(1) - 1) + topk_probs[0, i].item()) / seq.size(1)

                # correct scoring: accumulate log probabilities
                new_score = score + topk_probs[0, i].item()

                new_beams.append((new_seq, new_score))

        # keep only top beam_size candidates for next step
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if len(completed) >= beam_size:
            break

    # add any unfinished beams to completed pool
    completed += beams

    # length-normalize final scores before selecting best sequence
    best_seq = sorted(
        completed,
        key=lambda x: x[1] / x[0].size(1),
        reverse=True
    )[0][0]

    return best_seq.squeeze(0)  # (seq_len,)