from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import ROOT


def get_all_sentences(ds, lang):
    # generator function — yields one sentence at a time from the dataset
    # using a generator instead of a list avoids loading the entire corpus
    # into memory at once, which matters for large datasets
    # ds:   HuggingFace dataset object
    # lang: language code string e.g. 'en', 'fr', 'de' for us it will be 'mr'
    for item in ds:
        yield item['translation'][lang]


def build_tokenizer(config, ds, lang):
    # check if a pre-trained tokenizer already exists on disk
    # training a BPE tokenizer is slow — we save it after the first run
    # and reload it on subsequent runs to avoid retraining every time
    tokenizer_path = ROOT / config['tokenizer_file'].format(lang)

    if not tokenizer_path.exists():
        # BPE (Byte Pair Encoding) tokenizer — starts with individual characters
        # and iteratively merges the most frequent pairs into subword units
        # this handles unknown words gracefully since any word can be broken
        # into known sub-pieces. unk_token handles truly unseen sub-pieces
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

        # whitespace pre-tokenizer splits raw text into words first
        # BPE then operates within each word to find subword boundaries
        tokenizer.pre_tokenizer = Whitespace()

        # special tokens every NLP model needs:
        # [UNK] — unknown token for out-of-vocabulary sub-pieces
        # [PAD] — padding token to make sequences equal length in a batch
        # [SOS] — start of sequence, signals decoder to begin generating
        # [EOS] — end of sequence, signals decoder to stop generating
        # min_frequency=2 — a subword pair must appear at least twice
        # to be merged, filters out noise from rare character combinations
        trainer = BpeTrainer(
            special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'],
            min_frequency=2
        )

        # train BPE on the raw sentence stream for this language
        # get_all_sentences is a generator so memory stays low during training
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        # save trained tokenizer to disk as JSON for reuse
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer trained and saved to {tokenizer_path}")

    else:
        # reload previously trained tokenizer from disk
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Tokenizer loaded from {tokenizer_path}")

    return tokenizer