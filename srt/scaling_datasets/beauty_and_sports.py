from srt import fs, settings
from srt.serialization import PartitionedIterable
from .scaled_datasets import create_recbole_dataset


def build_beauty_and_sports_datasets():
    meta_pi = PartitionedIterable(settings.TOKENIZED_METADATA_DIR).multicore().add_progress(desc='load metadata')
    all_meta = {e['asin']: e for e in meta_pi}

    pi = PartitionedIterable([settings.RAW_BEAUTY_FNAME]).add_tfm(key_map)
    create_recbole_dataset(pi, all_meta, fs.join(settings.RECBOLE_DATASETS, 'amazon-beauty'))

    pi = PartitionedIterable([settings.RAW_SPORTS_FNAME]).add_tfm(key_map)
    create_recbole_dataset(pi, all_meta, fs.join(settings.RECBOLE_DATASETS, 'amazon-sports'))


def key_map(it):
    d = dict(
        reviewerID='user_id',
        asin='item_id',
        overall='rating',
        unixReviewTime='timestamp'
    )
    for doc in it:
        for key, val in d.items():
            doc[val] = doc.pop(key)
        yield doc
