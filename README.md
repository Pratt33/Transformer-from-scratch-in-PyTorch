# Transformer from Scratch in PyTorch
### English → Marathi Neural Machine Translation

Built as a deep dive into PyTorch and the Transformer architecture. The goal was twofold:
1. Understand how to implement any deep learning architecture in PyTorch from first principles
2. Build a reference for future PyTorch implementations — covering modules, training loops, GPU usage, and deployment

---

## What is a Transformer?

A Transformer is a type of neural network that uses an **attention mechanism** to build contextual understanding of sequences. Introduced in the 2017 paper *"Attention is All You Need"*, it has become the foundation of nearly every modern AI language system.

Its key advantage over earlier architectures like RNNs is that it processes the entire input **in parallel** rather than token by token. This makes it significantly faster to train, especially on GPUs, and allows it to capture long-range dependencies between words far more effectively.

---

## Architecture Overview

```
Input Tokens
     ↓
Token Embedding + Positional Encoding
     ↓
Encoder (N × EncoderBlock)
  └── Multi-Head Self-Attention
  └── Feed-Forward Block
  └── Add & Norm (Residual Connections)
     ↓
Decoder (N × DecoderBlock)
  └── Masked Multi-Head Self-Attention
  └── Cross-Attention (attends to Encoder output)
  └── Feed-Forward Block
  └── Add & Norm (Residual Connections)
     ↓
Projection Layer → vocab_size
     ↓
Output Translation
```

---

## PyTorch Concepts Covered

This project was as much about learning PyTorch patterns as it was about building the Transformer. Below is a reference for each major concept encountered.

### `nn.Module` — The Building Block of Everything

Every component in this project subclasses `nn.Module`. The contract is simple but important:
- Put all **learnable components** (Linear layers, Embeddings, etc.) in `__init__`
- Put all **computation** in `forward`

PyTorch uses this contract to automatically track parameters, move the model to a device with `.to(device)`, save/load weights with `state_dict()`, and switch between train/eval modes.

```python
class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)  # registered as a parameter

    def forward(self, x):
        return self.linear(x)              # computation lives here
```

Calling `model(inputs)` automatically routes to `model.forward(inputs)` — this is PyTorch's `__call__` mechanism.

### `nn.Parameter` vs `register_buffer`

| | Trained? | Saved with model? | Use case |
|---|---|---|---|
| `nn.Parameter` | ✅ Yes | ✅ Yes | Learnable weights |
| `register_buffer` | ❌ No | ✅ Yes | Positional encoding, fixed constants |

Positional encodings are registered as buffers — they travel with the model (e.g. when calling `.to(device)`) but are never updated by the optimizer.

### Device-Agnostic Code

Never hardcode `"cuda"`. Always write code that works on any hardware:

```python
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model = Transformer(...).to(device)

# move data in the training loop
src = src.to(device, non_blocking=True)
```

`non_blocking=True` enables asynchronous CPU→GPU transfers when `pin_memory=True` is set on the DataLoader — a meaningful throughput improvement.

### The 5-Step Training Update Cycle

Every PyTorch training loop follows this exact order:

```python
optimizer.zero_grad()                          # 1. clear old gradients
loss = criterion(output, target)               # 2. compute loss
loss.backward()                                # 3. compute new gradients via backprop
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 4. clip gradients
optimizer.step()                               # 5. update parameters
```

Gradients **accumulate** by default in PyTorch — `zero_grad()` must be called every step or gradients from the previous batch contaminate the current one. This is a feature (enabling gradient accumulation) but a common source of bugs if forgotten.

### Mixed Precision Training

Runs the forward pass in `float16` where safe, falling back to `float32` where needed. Halves memory usage and significantly speeds up training on modern GPUs.

```python
scaler = GradScaler(enabled=(device.type == "cuda"))

with autocast(device_type=device.type, enabled=(device.type == "cuda")):
    output = model(src, tgt)
    loss   = criterion(output, target)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)                     # unscale BEFORE clipping
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Important: always call `scaler.unscale_()` before `clip_grad_norm_()` — gradients must be in their true magnitude for the clip threshold to be meaningful.

### Dataset and DataLoader

```python
class BilingualDataset(Dataset):
    def __len__(self):   ...   # total number of samples
    def __getitem__(self, idx): ...  # one sample by index

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,       # parallel data loading — frees GPU from waiting
    pin_memory=True      # faster CPU→GPU transfers
)
```

`num_workers` and `pin_memory` are the two most impactful DataLoader settings for GPU training. A GPU sitting at low utilization often means the data pipeline is the bottleneck, not the model.

### Checkpointing — Save and Resume Training

```python
# save — always include optimizer state to resume correctly
torch.save({
    "epoch":                epoch,
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "global_step":          global_step,
}, filepath)

# load — map_location handles GPU→CPU and CPU→GPU transfers
state = torch.load(filepath, map_location=device)
model.load_state_dict(state['model_state_dict'])
optimizer.load_state_dict(state['optimizer_state_dict'])
```

Saving optimizer state preserves Adam's momentum statistics — without this, training resumes as if from scratch rather than continuing smoothly.

---

## Component Walkthrough

### 1. Token Embedding — `embedding.py`

Converts integer token IDs into dense vectors via a trainable lookup table of shape `(vocab_size, d_model)`.

- Input: `(batch_size, seq_len)` — integer token IDs
- Output: `(batch_size, seq_len, d_model)`

Vectors are scaled by `√d_model` after lookup. This brings embedding magnitudes up to be comparable with positional encoding values (which are bounded between -1 and 1), preventing either from dominating when they are added together.

### 2. Positional Encoding — `encoding.py`

Transformers process all tokens in parallel and have no inherent sense of order. Positional encoding solves this by adding a position-dependent signal to each token's embedding.

Fixed sinusoidal functions are used (not learned). Each position gets a unique "fingerprint" across `d_model` dimensions — high-frequency dimensions capture fine-grained local position differences, low-frequency dimensions capture coarse global relationships.

- Registered as a `buffer` — not trained, but saved with the model and moved to device automatically
- `detach()` used when adding to embeddings to prevent gradients flowing through it

Output shape: `(batch_size, seq_len, d_model)` — unchanged, position information now baked in.

### 3. Multi-Head Attention — `attention.py`

The core of the Transformer. Allows every token to attend to every other token and build a contextual representation based on content.

**Query, Key, Value:** Each token projects into three roles simultaneously:
- **Query** — what this token is looking for
- **Key** — what this token advertises it contains  
- **Value** — what this token gives away when attended to

Attention weights are computed as:
```
attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Dividing by `√d_k` prevents dot products from growing large as dimensionality increases, which would push softmax into regions with near-zero gradients and cause training to stall.

**Multi-head:** Rather than running attention once in the full `d_model` space, it runs `h` times in parallel in `d_k = d_model/h` dimensional subspaces. Each head can learn to attend to different types of relationships (syntactic, semantic, positional). Outputs are concatenated and projected back to `d_model` via the output projection `W_o`.

**Shape flow:**
```
(B, S, d_model) → split heads → (B, h, S, d_k) → attention → (B, h, S, d_k) → merge → (B, S, d_model)
```

**Important:** In cross-attention (decoder attending to encoder), Q comes from the decoder and K, V come from the encoder — they can have different sequence lengths. The output sequence length always matches the Query sequence length.

### 4. Layer Normalization — `normalization.py`

Normalizes each token's vector independently across the `d_model` dimension, then applies learned scale (`alpha`) and shift (`bias`) parameters. This is applied after every attention and feed-forward sublayer.

**Why Layer Norm over Batch Norm?** Layer Norm normalizes per token independently of batch size and sequence length, making it stable for variable-length sequences and small batches — the standard choice for Transformers.

### 5. Feed-Forward Block — `feedforward.py`

Applied independently to each token after attention. Two linear projections with a GELU activation:

```
d_model → d_ff → d_model    (d_ff = 4 × d_model = 2048)
```

The expansion to `d_ff` gives the model capacity to learn complex non-linear transformations in the token dimension. Without the non-linearity, two linear layers would collapse into one.

### 6. Residual Connections — `residual.py`

Wraps every sublayer (attention or feed-forward) with:
```
x = x + dropout(sublayer(LayerNorm(x)))
```

This is the Pre-Norm pattern (normalize before sublayer). The residual addition solves the vanishing gradient problem — gradients can flow directly back through the addition without passing through the sublayer, keeping signal strong in deep networks.

### 7. Encoder

Stacks `N=6` EncoderBlocks. Each block applies self-attention followed by a feed-forward block, both with residual connections. A final LayerNorm is applied after all blocks.

The encoder runs once per input and its output is reused at every decoder step during inference.

### 8. Decoder

Stacks `N=6` DecoderBlocks. Each block has three sublayers:
1. **Masked self-attention** — decoder tokens attend to each other, but the causal mask prevents attending to future positions
2. **Cross-attention** — Q from decoder, K and V from encoder output. This is how the decoder reads the source sentence
3. **Feed-forward** — same as encoder

### 9. Projection Layer — `projection.py`

Maps the decoder output from `d_model` to `vocab_size`, producing a score for every token in the target vocabulary. `log_softmax` converts scores to log probabilities.

---

## Masking

Two types of masks are used:

**Padding mask** — marks `[PAD]` tokens so attention ignores them. Shape `(B, 1, 1, S)` — broadcasts across heads and query positions.

**Causal mask** — lower triangular matrix that prevents the decoder from attending to future token positions during training. Without this, the model would "see the answer" during training and fail at inference.

```python
def causal_mask(size):
    return torch.tril(torch.ones(1, size, size)).bool()
```

---

## Training

**Tokenization:** BPE (Byte Pair Encoding) via HuggingFace `tokenizers`. Trained separately for source and target languages on training data only — never on validation or test data (data leakage).

**Dataset:** OPUS-100 English-Marathi pairs. Each sample returns:
- `encoder_input` — `[SOS] + tokens + [EOS] + [PAD]...`
- `decoder_input` — `[SOS] + tokens + [PAD]...` (teacher forcing input)
- `label` — `tokens + [EOS] + [PAD]...` (what decoder should predict)

**Teacher forcing:** During training the decoder receives ground truth previous tokens as input rather than its own predictions. This makes training stable and fast. During inference there is no ground truth — the model feeds its own output back as input.

**Loss:** `CrossEntropyLoss` with `ignore_index=[PAD]` and `label_smoothing=0.1`. Label smoothing prevents the model from becoming overconfident — the true label receives probability 0.9 with 0.1 spread across all other tokens.

**Optimizer:** Adam with `eps=1e-9` and `lr=1e-4`. Xavier uniform initialization for all weight matrices with `dim > 1`.

---

## Inference

**Greedy decoding:** At each step, pick the single highest-probability token. Fast and deterministic but can miss better overall sequences.

**Beam search:** Keep the top `k` candidate sequences at every step, scoring by length-normalized log probability. Slower but produces higher quality translations.

```
Greedy:      [SOS] → w1 → w2 → w3 → [EOS]   (1 candidate at each step)
Beam (k=4):  [SOS] → top-4 → top-4 → top-4 → best overall sequence
```

---

## Evaluation Metrics

| Metric | What it measures | Better when |
|--------|-----------------|-------------|
| BLEU | N-gram overlap with reference translation | Higher |
| CER | Character-level edit distance | Lower |
| WER | Word-level edit distance | Lower |

---

## Deployment

- **Gradio** (`app.py`) — web UI with text input/output, shareable link
- **FastAPI** (`api.py`) — REST API, accepts POST `/translate` with JSON body
- Both share the same `loader.py` which loads the model **once at startup** — never per request

Live demo: [huggingface.co/spaces/Pratt333/english-to-marathi](https://huggingface.co/spaces/Pratt333/english-to-marathi)

---

## Project Structure

```
transformer/
├── model/
│   ├── embedding/
│   │   ├── embedding.py       # TokenEmbedding
│   │   └── encoding.py        # PositionalEncoding
│   ├── transformer/
│   │   ├── attention.py       # MultiHeadAttention
│   │   └── feedforward.py     # FeedForwardBlock
│   ├── common/
│   │   ├── normalization.py   # LayerNorm
│   │   └── residual.py        # ResidualConnection
│   ├── encoder/
│   │   ├── encoderblock.py    # EncoderBlock
│   │   └── encoder.py         # Encoder
│   ├── decoder/
│   │   ├── decoderblock.py    # DecoderBlock
│   │   ├── decoder.py         # Decoder
│   │   └── projection.py      # ProjectionLayer
│   └── main.py                # build_transformer()
├── tokenizer/
│   ├── tokenizer.py           # BPE tokenizer training
│   ├── dataset.py             # BilingualDataset
│   └── dataloader.py          # get_ds()
├── app/
│   ├── app.py                 # Gradio interface
│   ├── api.py                 # FastAPI endpoints
│   └── loader.py              # Model loading + translate()
├── evaluation/
│   ├── inference.py           # greedy_decode, beam_search_decode
│   └── validation.py          # validate_model with metrics
├── main.py                    # Training loop
├── config.py                  # Hyperparameters + device + checkpoint paths
└── requirements.txt
```

---

## Key Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `d_model` | 512 | Embedding dimension throughout |
| `N` | 6 | Number of encoder and decoder blocks |
| `h` | 8 | Number of attention heads |
| `d_ff` | 2048 | Feed-forward inner dimension (4 × d_model) |
| `seq_len` | 350 | Maximum sequence length |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 8 | Sentences per training step |
| `lr` | 1e-4 | Adam learning rate |

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers)
- [OPUS-100 Dataset](https://huggingface.co/datasets/Helsinki-NLP/opus-100)