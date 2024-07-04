import re

import sentencepiece as spm

from srt import fs, settings

wp = re.compile('\w+')


class SentencePieceTokenizer:
    def __init__(self, fname):
        self.fname = fname
        self.tokenizer = spm.SentencePieceProcessor(model_file=fname)

    @staticmethod
    def train(text_iterator, out_dir):
        train_fname = fs.join(settings.TMP_DIR, 'sentence_piece_training.txt')
        with open(train_fname, 'w') as f:
            for i, text in enumerate(text_iterator):
                if i > 0: f.write('\n')
                to_write = _normalize(text.lower())
                if to_write:
                    f.write(to_write)

        spm.SentencePieceTrainer.train(
            input=train_fname, model_prefix='m', vocab_size=30000, model_type='bpe'
        )
        fs.move('m.model', fs.ensure_exists(out_dir))
        fs.move('m.vocab', out_dir)

    def tokenize(self, text, max_tokens):
        text = _normalize(text)
        return ' '.join(self.tokenizer.encode(text, out_type=str)[:max_tokens])


def _normalize(title):
    return ' '.join(wp.findall(title.lower()))