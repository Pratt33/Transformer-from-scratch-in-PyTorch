import torch
from pathlib import Path

# device selection — automatically picks the best available hardware
# cuda: NVIDIA GPU, mps: Apple Silicon, cpu: fallback
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# resolves to the directory where config.py lives — always the project root
# regardless of which subfolder the script is run from
ROOT = Path(__file__).parent

def get_config():
    return {
        # --- Training Dynamics ---
        'batch_size': 8,          # number of sentence pairs processed per update step
        'num_epochs': 20,         # full passes over the training dataset
        'lr': 10**-4,             # learning rate — how large each parameter update is
        'dropout': 0.1,           # fraction of neurons randomly zeroed during training

        # --- Architecture ---
        # these must stay identical between training and inference
        # changing any of these invalidates saved checkpoints
        'seq_len': 350,           # max token length — sequences truncated or padded to this
        'd_model': 512,           # embedding dimension throughout the model
        'd_ff': 2048,             # feedforward inner dimension — 4 * d_model by convention
        'h': 8,                   # number of attention heads — d_model must be divisible by h
        'N': 6,                   # number of encoder and decoder blocks stacked

        # --- Language Pair ---
        'lang_src': 'en',         # source language code
        'lang_tgt': 'mr',         # target language code (Marathi)
                                  # opus-100 has this pair

        # --- Checkpointing ---
        'model_folder': 'weights',       # directory where .pt checkpoint files are saved
        'model_basename': 'tmodel_',     # prefix for checkpoint filenames e.g. tmodel_0.pt
        'preload': 'latest',                 # None = start fresh
                                         # 'latest' = resume from most recent checkpoint
                                         # '5' = resume from epoch 5 specifically

        # --- Logging ---
        'tokenizer_file': 'tokenizer_{0}.json',   # {0} is replaced by language code
        'experiment_name': 'runs/tmodel'          # TensorBoard watches this directory
    }


def get_weights_file_path(config, epoch):
    # constructs full path for a specific epoch's checkpoint
    # e.g. epoch=3 → ./weights/tmodel_3.pt
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(ROOT / config['model_folder'] / model_filename)


def get_latest_weights_file_path(config):
    # finds the most recently saved checkpoint automatically
    # used when config['preload'] == 'latest' to resume training
    # without needing to manually specify the epoch number
    weights_dir = ROOT / config['model_folder']
    weights_files = list(weights_dir.glob(f"{config['model_basename']}*.pt"))
    if not weights_files:
        # no checkpoints exist yet — training will start from scratch
        return None
    # sort alphabetically — tmodel_0, tmodel_1 ... tmodel_19
    weights_files.sort(key=lambda x: int(x.stem.split("_")[1]))
    return str(weights_files[-1])