import tempfile
from itertools import islice
from multiprocessing import Process
from threading import Thread

import requests

from srt import fs, settings
from srt.progress import progress
from srt.serialization import iter_jl
from srt.serialization.read import iter_csv
from srt.serialization.write import write_partitioned
from srt.time_it import timeit


def download_raw_files():
    processes = [
        download_file_in_background(
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz',
            settings.RAW_BEAUTY_FNAME, binary=True
        ),
        download_file_in_background(
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz',
            settings.RAW_SPORTS_FNAME, binary=True
        ),
        download_file_in_background(
            "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/all_csv_files.csv",
            settings.RAW_RATINGS_FNAME
        ),

        download_file_in_background(
            'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles/All_Amazon_Meta.json.gz',
            settings.RAW_METADATA_FNAME, binary=True
        )
    ]
    for p in processes:
        p.join()


def partition_big_files():
    with timeit('partition ratings'):
        partition_ratings()

    with timeit('partition metadata'):
        partition_metadata()


def partition_ratings(partition_size=7_000_000):
    raw_data_iter = iter_csv(settings.RAW_RATINGS_FNAME, fieldnames=['item_id', 'user_id', 'rating', 'timestamp'])
    data_iter = iter(progress(_parse_stars(raw_data_iter), desc='partition ratings'))
    fs.ensure_clean(settings.PARTITIONED_RATINGS_DIR)

    if settings.ONLY_FEW:
        partition_size = 700_000
        data_iter = islice(data_iter, 0, 3_500_000)

    write_partitioned(data_iter, settings.PARTITIONED_RATINGS_DIR, partition_size=partition_size)


def partition_metadata(partition_size=700_000):
    data_iter = progress(_reduce_metadata_fields(iter_jl(settings.RAW_METADATA_FNAME)), desc='partition metadata')
    fs.ensure_clean(settings.PARTITIONED_METADTA_DIR)

    if settings.ONLY_FEW:
        partition_size = 70_000
        data_iter = islice(data_iter, 0, 700_000)

    write_partitioned(data_iter, settings.PARTITIONED_METADTA_DIR, partition_size=partition_size)


def _reduce_metadata_fields(it):
    fields = ['title', 'brand', 'category', 'asin']
    for doc in it:
        doc = {f: doc.get(f) for f in fields}
        if 'getTime' in doc['title']: continue
        if not doc['title'].strip(): continue

        categories = [e for e in doc.pop('category', []) if len(e) < 40][:3]

        for i, c in enumerate((categories + [None, None, None])[:3]):
            doc[f'category_l{i + 1}'] = c

        yield doc


def download_file_in_background(url, fname, binary=False, chunk_size=20*1024*1024):
    print(f'downloading {url} into {fs.name(fname)}')
    t = Process(target=download_file, args=(url, fname, binary, chunk_size))
    t.start()
    return t


def download_file(url, fname, binary=False, chunk_size=20*1024*1024):
    try:
        tmp_fname = tempfile.mktemp(dir=settings.TMP_DIR)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(tmp_fname, 'wb' if binary else 'w') as f:
                tot = int(r.headers['Content-Length']) // chunk_size
                data_iter = r.iter_content(chunk_size=chunk_size)
                desc = f'download into {fs.name(fname)} (tmp file: {tmp_fname})'
                for chunk in progress(data_iter, tot=tot, desc=desc):
                    if not binary:
                        chunk = chunk.decode('utf8')
                    f.write(chunk)

        fs.move(tmp_fname, fname)
    finally:
        fs.ensure_not_exists(tmp_fname)


def _parse_stars(it):
    for doc in it:
        doc['rating'] = int(doc['rating'][:1])
        yield doc
