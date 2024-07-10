from srt import fs, settings
from srt.progress import progress
from srt.serialization import JlWriter, PartitionedIterable
from .sentence_piece_tokenizer import SentencePieceTokenizer


def _text_iterator():
    meta_pi = PartitionedIterable(settings.PARTITIONED_METADTA_DIR).multicore().add_progress(desc='load metadata')
    if settings.ONLY_FEW: meta_pi = meta_pi.set_limit(100_000)

    for i, meta in enumerate(meta_pi):
        yield meta['title']
        yield meta['brand']


def train_sentence_piece():
    SentencePieceTokenizer.train(_text_iterator(), settings.SENTENCE_PIECE_DIR)


def tokenize_metadata():
    meta_pi = PartitionedIterable(settings.PARTITIONED_METADTA_DIR)
    meta_pi.pmap(_tokenize_metadata, out_dir=fs.ensure_clean(settings.TOKENIZED_METADATA_DIR))


def _tokenize_metadata(it, out_dir):
    tokenizer = SentencePieceTokenizer(fs.join(settings.SENTENCE_PIECE_DIR, 'm.model'))
    with JlWriter(fs.join(out_dir, it.name)) as writer:
        metadata_iter = progress(it, desc=f'tokenize {fs.name(fs.parent(it.path))}/{it.name}')
        for item_meta in metadata_iter:
            # 99% percentile of title length, the other 1% are gigantic and causes a huge GPU memory footprint
            item_meta['tokenized_title'] = tokenizer.tokenize(item_meta['title'], max_tokens=40)
            b = item_meta['tokenized_brand'] = tokenizer.tokenize(item_meta['brand'], max_tokens=8)
            if not b: item_meta['tokenized_brand'] = 'unknown'

            writer.write_doc(item_meta)
