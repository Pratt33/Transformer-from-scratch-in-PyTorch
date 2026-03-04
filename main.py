import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
from torch.amp import autocast, GradScaler
from model.main import build_transformer
from tokenizer.dataloader import get_ds
from config import device, get_config, get_weights_file_path, get_latest_weights_file_path


def get_model(cfg, vocab_src_len, vocab_tgt_len):
    # constructs the full Transformer and move it to the correct device
    model = build_transformer(
        vocab_src_len, vocab_tgt_len,
        cfg['seq_len'], cfg['seq_len'],
        cfg['d_model'], cfg['N'],
        cfg['h'], cfg['dropout'], cfg['d_ff']
    )
    return model


def train_model(cfg):
    # cudnn.benchmark: lets cuDNN auto-tune convolution algorithms for your hardware
    # small one-time cost at startup, meaningful throughput gain during training
    torch.backends.cudnn.benchmark = True
    # create weights directory if it doesn't exist
    # parents=True: creates intermediate dirs, exist_ok=True: no error if already exists
    Path(cfg['model_folder']).mkdir(parents=True, exist_ok=True)

    tokenizer_src, tokenizer_tgt, train_dataloader, val_dataloader, _ = get_ds(cfg)
    model = get_model(cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # torch.compile: traces the model and compiles it to optimized kernels
    # significant speedup on CUDA — skipped on CPU/MPS where it has no benefit
    if device.type == "cuda":
        model = torch.compile(model)

    # TensorBoard writer — logs scalars to experiment_name directory
    # run 'tensorboard --logdir runs' in terminal to visualize loss curves
    writer   = SummaryWriter(log_dir=cfg['experiment_name'])
    optimizer = Adam(model.parameters(), lr=cfg['lr'], eps=1e-9)

    # GradScaler: scales loss upward before backward() to prevent fp16 underflow
    # automatically skips optimizer step if gradients contain inf/nan
    # enabled=False on CPU/MPS — scaler becomes a no-op, code stays device-agnostic
    scaler = GradScaler(enabled=(device.type == "cuda"))

    initial_epoch = 0
    global_step   = 0

    # checkpoint loading — restores full training state so training resumes
    # exactly where it left off including epoch count and optimizer momentum
    if cfg['preload'] == 'latest':
        model_filename = get_latest_weights_file_path(cfg)
    elif cfg['preload']:
        model_filename = get_weights_file_path(cfg, cfg['preload'])
    else:
        model_filename = None

    if model_filename:
        # map_location=device ensures checkpoints saved on GPU load correctly on CPU
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step   = state['global_step']
        print(f"Resuming from epoch {initial_epoch}, global step {global_step}")

    # CrossEntropyLoss expects raw logits — so not apply softmax before this
    # ignore_index=PAD: padding positions don't contribute to loss
    # label_smoothing=0.1: prevents overconfidence — true label gets 0.9,
    # remaining 0.1 spread across all other tokens. improves generalization
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, cfg['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch: {epoch:02d}")

        for batch in batch_iterator:
            # move all batch tensors to device in one place
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)  # (B, seq_len)
            encoder_mask  = batch['encoder_mask'].to(device, non_blocking=True)   # (B, 1, 1, seq_len)
            decoder_mask  = batch['decoder_mask'].to(device, non_blocking=True)   # (B, 1, seq_len, seq_len)
            label         = batch['label'].to(device, non_blocking=True)          # (B, seq_len)

            optimizer.zero_grad()

            # autocast: runs forward pass in fp16 where safe, fp32 where needed
            # device_type must be a string — use device.type not device itself
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # forward pass through the three stages of the Transformer
                encoder_output = model.encode(encoder_input, encoder_mask)         # (B, seq_len, d_model)
                decoder_output = model.decode(decoder_input, encoder_output,
                                              encoder_mask, decoder_mask)          # (B, seq_len, d_model)
                proj_output    = model.project(decoder_output)                     # (B, seq_len, vocab_size)

                # reshape for CrossEntropyLoss which expects (N, C) and (N,)
                # where N = B * seq_len and C = vocab_size
                loss = loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()),  # (B*seq_len, vocab_size)
                    label.view(-1)                                          # (B*seq_len,)
                )

            # backward through scaled loss
            scaler.scale(loss).backward()

            # unscale BEFORE clipping — gradients must be in their true magnitude
            # for the clip threshold to be meaningful
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # step only updates if gradients are finite — skips bad batches automatically
            scaler.step(optimizer)
            scaler.update()

            # display live loss in tqdm progress bar
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # log to TensorBoard — flush ensures it writes immediately
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            global_step += 1

        # always save uncompiled weights — portable across compiled and non-compiled inference
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

        # save full training state after every epoch
        # saving optimizer state allows resuming with correct momentum/adam statistics
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step":          global_step,
        }, get_weights_file_path(cfg, f"{epoch:02d}"))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)