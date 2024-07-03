import pickle as pkl
from collections import Counter
from itertools import chain

from src import fs, settings
from src.progress import progress
from src.serialization import iter_jl, PartitionedIterable, JlWriter
from src.time_it import timeit


def post_process_ratings():
    # TODO: compute everything in one pass
    with timeit('remove ratings with no title'):
        remove_ratings_with_no_title(settings.PARTITIONED_RATINGS_DIR, fs.join(settings.TMP_DIR, 'ratings_with_meta'))

    with timeit('filter cold users'):
        filter_cold_users(
            fs.join(settings.TMP_DIR, 'ratings_with_meta'),
            fs.join(settings.TMP_DIR, 'ratings_no_cold_users')
        )

    with timeit('filter_beauty_and_sports_users'):
        filter_beauty_and_sports_users(
            fs.join(settings.TMP_DIR, 'ratings_no_cold_users'),
            settings.POST_PROCESSED_RATINGS_DIR
        )


def remove_ratings_with_no_title(ratings_dir, out_dir):
    meta_pi = PartitionedIterable(settings.PARTITIONED_METADTA_DIR).add_progress(desc='load metadata').multicore()
    # if settings.ONLY_FEW: 
    #     meta_pi = meta_pi.set_limit(100_000)
    all_item_ids = set(e['asin'] for e in meta_pi)
    fname = fs.join(settings.TMP_DIR, 'items_with_meta.pkl')
    with open(fname, 'wb') as f:
        pkl.dump(all_item_ids, f, 2)

    # TODO: refactor this with _filter_user_set
    ratings_pi = PartitionedIterable(ratings_dir)
    ratings_pi.pmap(_remove_no_meta_items, out_dir=fs.ensure_clean(out_dir), item_set_fname=fname)


def filter_cold_users(ratings_dir, out_dir):
    distr = get_events_per_user(ratings_dir)
    user_set = set(user_id for user_id, cnt in distr.items() if cnt >= 5)
    print(f'Only {len(user_set)} have 5 interactions or more ({len(user_set) / len(distr) * 100:.02f}%')

    with open(settings.WARM_USER_SET_FNAME, 'wb') as f:
        pkl.dump(user_set, f, 2)

    PartitionedIterable(ratings_dir).pmap(
        _filter_user_set, out_dir=fs.ensure_clean(out_dir), user_set_fname=settings.WARM_USER_SET_FNAME
    )


def filter_beauty_and_sports_users(ratings_dir, out_dir):
    data_iter = chain(iter_jl(settings.RAW_BEAUTY_FNAME), iter_jl(settings.RAW_SPORTS_FNAME))
    user_set = set(e['reviewerID'] for e in progress(data_iter, desc='load beauty and sports users'))

    fname = fs.join(settings.TMP_DIR, 'beauty_and_sports_users.pkl')
    with open(fname, 'wb') as f:
        pkl.dump(user_set, f)

    PartitionedIterable(ratings_dir).pmap(_filter_user_set, out_dir=fs.ensure_clean(out_dir),
                                          user_set_fname=fname, keep=False)


def _remove_no_meta_items(it, out_dir, item_set_fname):
    with open(item_set_fname, 'rb') as f:
        item_set = pkl.load(f)

    with JlWriter(fs.join(out_dir, it.name)) as writer:
        removed = 0
        total = 0
        for doc in it:
            total += 1
            if doc['item_id'] not in item_set:
                removed += 1
                continue

            writer.write_doc(doc)

    with progress.lock:
        print(f'Removed {removed} ({removed / total * 100:.02f}%) rows from {it.name}, they have no meta')


def _user_distr(it):
    res = Counter()
    for doc in it:
        res[doc['user_id']] += 1
    return res


def get_events_per_user(ratings_dir):
    pi = PartitionedIterable(ratings_dir)
    distr = Counter()

    for partial_distr in pi.pmap(_user_distr, results_iterator=True):
        distr.update(partial_distr['res'])

    return distr


def _filter_user_set(it, out_dir, user_set_fname, keep=True):
    """
    keep=True means only keep the users in the set
    keep=False means discard the users in the set
    """

    with open(user_set_fname, 'rb') as f:
        user_set = pkl.load(f)

    with JlWriter(fs.join(out_dir, it.name)) as writer:
        removed = 0
        total = 0
        for doc in it:
            total += 1
            if keep and doc['user_id'] not in user_set:
                removed += 1
                continue
            elif not keep and doc['user_id'] in user_set:
                removed += 1
                continue

            writer.write_doc(doc)

    with progress.lock:
        print(f'Removed {removed} ({removed / total * 100:.02f}%) rows from {it.name}')
