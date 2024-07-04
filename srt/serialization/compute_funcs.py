import heapq
import ujson as json

import numpy as np
import psutil

from app import fs
from .write import PathHasher, write_jl
from .iterables import PartitionedIterable


def partition_by(it, out_dir, key, n_buckets):
    fs.ensure_exists(out_dir)
    with PathHasher(out_dir, n_buckets) as out_hasher:
        for doc in it:
            writer = out_hasher.get_writer(multiget(doc, key), it.name)
            writer.write_doc(doc)


def multiget(doc, k):
    if isinstance(k, str):
        return doc[k]
    else:  # es iterable
        return tuple(doc[sk] for sk in k)


def sort_by(it, out_dir, key, batch_size):
    tmp_dir = fs.join(out_dir, '_tmp', it.name)
    fs.ensure_clean(tmp_dir)

    curr_batch = 0
    batch = []
    for doc in it:
        batch.append(doc)

        if len(batch) == batch_size:
            batch.sort(key=lambda x: multiget(x, key))
            write_jl(batch, fs.join(tmp_dir, str(curr_batch) + '.jl.gz'))
            del batch
            batch = []
            curr_batch += 1

    if batch:
        batch.sort(key=lambda x: multiget(x, key))
        write_jl(batch, fs.join(tmp_dir, str(curr_batch) + '.jl.gz'))
        del batch
        curr_batch += 1

    all_iters = PartitionedIterable(tmp_dir).sub_iters()
    ordered_events = heapq.merge(*all_iters, key=lambda x: multiget(x, key))
    write_jl(ordered_events, fs.join(out_dir, it.name + '.jl.gz'))
    fs.rmtree(tmp_dir)


def get_record_length(it):
    l = []
    for doc in it:
        l.append(len(json.dumps(doc)))
    return {'avg': np.mean(l), 'std': np.std(l)}


def estimate_batch_size(it):
    if it.limit is None or it.limit > 10000:
        it = it.set_limit(10000)

    lengths = it.pmap(get_record_length)
    m_length = np.mean([e['res']['avg'] for e in lengths])
    s_length = np.mean([e['res']['std'] for e in lengths])

    available_memory = psutil.virtual_memory().total * 0.6 / psutil.cpu_count()

    return int(available_memory / (m_length + 2 * s_length))
