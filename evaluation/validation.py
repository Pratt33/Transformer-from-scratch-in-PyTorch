from itertools import chain
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
import torch
from main import get_model
from tokenizer.dataloader import get_ds
from config import device, get_config, get_latest_weights_file_path
from inference import greedy_decode, beam_search_decode


def validate_model(model_instance, validation_ds, tokenizer_tgt_inst, max_len,
                   device_inst, print_msg, num_examples=10):
    model_instance.eval()

    # torchmetrics objects accumulate predictions across batches
    # .compute() at the end returns the final metric over all examples
    bleu = BLEUScore()
    cer  = CharErrorRate()
    wer  = WordErrorRate()

    count = 0

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch['encoder_input'].to(device_inst)
            encoder_mask  = batch['encoder_mask'].to(device_inst)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for decoding"

            # model_out = greedy_decode(
            #     model_instance, encoder_input, encoder_mask,
            #     tokenizer_tgt_inst, max_len, device_inst
            # )

            # use beam search instead for better translation quality
            model_out = beam_search_decode(
                model_instance, encoder_input, encoder_mask,
                tokenizer_tgt_inst, max_len, device_inst, beam_size=4
            )

            source_text    = batch['src_text'][0]
            target_text    = batch['tgt_text'][0]
            predicted_text = tokenizer_tgt_inst.decode(
                model_out.detach().cpu().numpy()
            )

            # torchmetrics BLEU expects list of predictions and list of list of references
            # BLEU: measures n-gram overlap between prediction and reference
            # CER:  character-level edit distance — good for morphologically rich languages
            # WER:  word-level edit distance — standard for translation quality
            bleu.update([predicted_text], [[target_text]])
            cer.update(predicted_text, target_text)
            wer.update(predicted_text, target_text)

            print_msg('-' * 80)
            print_msg(f"SOURCE:    {source_text}")
            print_msg(f"TARGET:    {target_text}")
            print_msg(f"PREDICTED: {predicted_text}")

            count += 1
            if count == num_examples:
                break

    # compute final metrics over all accumulated examples
    print_msg("=" * 80)
    print_msg(f"BLEU Score:        {bleu.compute():.4f}")
    print_msg(f"Char Error Rate:   {cer.compute():.4f}")
    print_msg(f"Word Error Rate:   {wer.compute():.4f}")


if __name__ == '__main__':
    cfg = get_config()
    tokenizer_src, tokenizer_tgt, train_dl, val_dl, test_dl = get_ds(cfg)
    model = get_model(
        cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    model_filename = get_latest_weights_file_path(cfg)
    if model_filename is None:
        raise FileNotFoundError("No trained model checkpoint found — train first")

    state = torch.load(model_filename, map_location=device)

    # handle checkpoints saved from torch.compile — strip _orig_mod. prefix if present
    state_dict = {
        k.replace('_orig_mod.', ''): v
        for k, v in state['model_state_dict'].items()
    }
    model.load_state_dict(state_dict)

    # chain combines val and test into one iterable without merging tensors
    valtest_dataloader = chain(val_dl, test_dl)

    validate_model(
        model, valtest_dataloader, tokenizer_tgt,
        cfg['seq_len'], device,
        print_msg=print,
        num_examples=10
    )