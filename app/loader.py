import torch
from tokenizers import Tokenizer
from config import ROOT, get_config, get_latest_weights_file_path, device
from main import get_model

# module-level variables — loaded once when this file is first imported
# any file that imports from here shares the same model instance in memory
_model      = None
_tokenizer_src = None
_tokenizer_tgt = None
_config     = None

def load_model():
    global _model, _tokenizer_src, _tokenizer_tgt, _config

    if _model is not None:
        # already loaded — return immediately, skip expensive reload
        return _model, _tokenizer_src, _tokenizer_tgt, _config

    _config = get_config()

    # load tokenizers directly from saved JSON files
    _tokenizer_src = Tokenizer.from_file(
        _config['tokenizer_file'].format(_config['lang_src'])
    )
    _tokenizer_tgt = Tokenizer.from_file(
        _config['tokenizer_file'].format(_config['lang_tgt'])
    )

    # build model architecture then load trained weights
    _model = get_model(
        _config,
        _tokenizer_src.get_vocab_size(),
        _tokenizer_tgt.get_vocab_size()
    ).to(device)

    model_path = get_latest_weights_file_path(_config)
    if model_path is None:
        raise FileNotFoundError("No checkpoint found — train the model first")

    state = torch.load(model_path, map_location=device)
    # handle checkpoints saved from torch.compile — strip _orig_mod. prefix if present
    state_dict = {
        k.replace('_orig_mod.', ''): v
        for k, v in state['model_state_dict'].items()
    }

    _model.load_state_dict(state_dict)

    # eval mode disables dropout — essential for deterministic inference
    _model.eval()
    print(f"Model loaded from {model_path}")

    return _model, _tokenizer_src, _tokenizer_tgt, _config


def translate(text: str) -> str:
    # single entry point for translation — used by both Gradio and FastAPI
    model, tokenizer_src, tokenizer_tgt, config = load_model()

    from evaluation.inference import greedy_decode

    # tokenize input and build encoder tensor
    tokens    = tokenizer_src.encode(text).ids
    seq_len   = config['seq_len']

    # truncate if needed, add [SOS] and [EOS], pad to seq_len
    sos_id = tokenizer_src.token_to_id('[SOS]')
    eos_id = tokenizer_src.token_to_id('[EOS]')
    pad_id = tokenizer_src.token_to_id('[PAD]')

    tokens     = tokens[:seq_len - 2]
    num_pad    = seq_len - len(tokens) - 2
    enc_input  = torch.tensor(
        [sos_id] + tokens + [eos_id] + [pad_id] * num_pad,
        dtype=torch.long
    ).unsqueeze(0).to(device)  # (1, seq_len)

    enc_mask = (enc_input != pad_id).unsqueeze(0).unsqueeze(0).bool().to(device)

    with torch.no_grad():
        output_ids   = greedy_decode(model, enc_input, enc_mask,
                                     tokenizer_tgt, seq_len, device)
        output_text  = tokenizer_tgt.decode(output_ids.cpu().numpy())

    return output_text