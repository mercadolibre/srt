import csv
import pickle as pkl
from random import Random

from srt import fs, settings
from srt.progress import progress
from srt.serialization import PartitionedIterable
from .post_process_ratings import _filter_user_set


def size2name(size):
    factor = 1_000_000 if size >= 1_000_000 else 1_000
    unit = 'M' if size >= 1_000_000 else 'K'

    return f'{size // factor}{unit}'


def build_scaling_datasets():
    sizes = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    if settings.ONLY_FEW:
        sizes = [100_000, 500_000]

    total_interactions = PartitionedIterable(settings.POST_PROCESSED_RATINGS_DIR).length

    with open(settings.WARM_USER_SET_FNAME, 'rb') as f:
        full_user_list = list(pkl.load(f))

    meta_pi = PartitionedIterable(settings.TOKENIZED_METADATA_DIR).add_progress(desc='load metadata').multicore()
    all_meta = {e['asin']: e for e in meta_pi}

    for target_size in progress(sizes, desc='build scaled datasets'):
        scaled_pi_path = _build_scaled_dataset(target_size, full_user_list, total_interactions)
        scaled_pi = PartitionedIterable(scaled_pi_path)
        dt_dir = fs.ensure_clean(fs.join(settings.RECBOLE_DATASETS, f'amazon-{size2name(target_size)}'))
        create_recbole_dataset(scaled_pi, all_meta, dt_dir)


def _build_scaled_dataset(target_size, full_user_list, total_interactions):
    out_dir = fs.ensure_clean(f'{settings.SCALED_DT_DIR}/{size2name(target_size)}')

    user_set = set(Random(42).sample(full_user_list, int(len(full_user_list) * target_size / total_interactions)))
    user_set_fname = fs.join(settings.SCALED_DT_DIR, f'user_set_{size2name(target_size)}.pkl')
    with open(user_set_fname, 'wb') as f:
        pkl.dump(user_set, f, 2)

    PartitionedIterable(settings.POST_PROCESSED_RATINGS_DIR).pmap(
        _filter_user_set, out_dir=out_dir, user_set_fname=user_set_fname
    )
    return out_dir


def _unique_items(it, item_id_key='item_id'):
    return set(e[item_id_key] for e in it)


def create_recbole_dataset(pi: PartitionedIterable, all_meta, out_dir):
    item_fieldnames = ['item_id:token', 'title:token_seq', 'category_l1:token', 'category_l2:token',
                       'category_l3:token', 'brand:token_seq']
    item_field_map = {
        'asin': 'item_id:token',
        'tokenized_title': 'title:token_seq',
        'category_l1': 'category_l1:token',
        'category_l2': 'category_l2:token',
        'category_l3': 'category_l3:token',
        'tokenized_brand': 'brand:token_seq'
    }

    inter_fieldnames = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
    inter_field_map = {field.split(':')[0]: field for field in inter_fieldnames}

    unique_items = set()

    for pres in pi.pmap(_unique_items):
        unique_items.update(pres['res'])

    print(f'dt has {len(unique_items)} items')

    fs.ensure_exists(out_dir)
    items_dropped = 0
    with open(fs.join(out_dir, fs.name(out_dir) + '.item'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=item_fieldnames, delimiter='\t')
        writer.writeheader()
        for item in progress(unique_items, desc='write items'):
            if item not in all_meta:
                items_dropped += 1
                continue

            raw_doc = all_meta[item]
            doc = {
                target_field: replace_tab(raw_doc[src_field])
                for src_field, target_field in sorted(item_field_map.items())
            }
            writer.writerow(doc)

    ratings_dropped = 0
    with open(fs.join(out_dir, fs.name(out_dir) + '.inter'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=inter_fieldnames, delimiter='\t')
        writer.writeheader()
        for raw_doc in pi.add_progress(desc='write inter'):
            if raw_doc['item_id'] not in all_meta:
                ratings_dropped += 1
                continue

            doc = {
                target_field: replace_tab(raw_doc[src_field])
                for src_field, target_field in sorted(inter_field_map.items())
            }
            writer.writerow(doc)

    if items_dropped:
        print(f'dropped {items_dropped} items')

    if ratings_dropped:
        print(f'dropped {ratings_dropped} ratings')


def replace_tab(s):
    if isinstance(s, str):
        return s.replace('\t', ' ')
    else:
        return s
