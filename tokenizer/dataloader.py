from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from dataset import BilingualDataset
from tokenizer import build_tokenizer


def get_ds(config):
    # load all three splits from HuggingFace Hub
    # each item: {"translation": {"en": "Hello", "fr": "Bonjour"}}
    raw_train_ds, raw_test_ds, raw_val_ds = load_dataset(
        'opus-100',
        f"{config['lang_src']}-{config['lang_tgt']}",
        split=['train', 'test', 'validation']
    )

    # build or reload tokenizers — both trained on raw training data only
    # test and val data must never influence vocabulary construction (data leakage)
    tokenizer_src = build_tokenizer(config, raw_train_ds, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, raw_train_ds, config['lang_tgt'])

    # compute max token lengths BEFORE wrapping in BilingualDataset
    # after wrapping, iterating returns tensors not raw translation dicts
    # useful for verifying seq_len config is large enough for the corpus
    max_len_src = 0
    max_len_tgt = 0
    for item in raw_train_ds:
        max_len_src = max(max_len_src, len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids))
        max_len_tgt = max(max_len_tgt, len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids))
    print(f"Max src token length: {max_len_src}")
    print(f"Max tgt token length: {max_len_tgt}")

    # wrap each raw split in BilingualDataset
    # each split gets its own Dataset instance — train/val/test never mix
    train_ds = BilingualDataset(raw_train_ds, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds   = BilingualDataset(raw_val_ds,   tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_ds  = BilingualDataset(raw_test_ds,  tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])

    # DataLoader wraps Dataset and handles batching, shuffling, and parallel loading
    # num_workers: load batches in parallel background processes — frees GPU from waiting
    # pin_memory:  stages batches in page-locked memory for faster CPU→GPU transfers
    #              only beneficial when training on GPU
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,             # shuffle training data each epoch
        num_workers=2,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,             # one sample at a time for greedy decoding during validation
        shuffle=False,            # never shuffle eval sets — results must be reproducible
        num_workers=2,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=config['batch_size'],
        shuffle=False,            # never shuffle test set
        num_workers=2,
        pin_memory=True
    )

    return tokenizer_src, tokenizer_tgt, train_dataloader, val_dataloader, test_dataloader