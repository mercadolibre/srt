import gzip
import json
import os
import tempfile
# from decimal import Decimal

import mmh3

from src import fs


class JlWriter:
    def __init__(self, fname, tmp_dir=None, use_tmp_file=True, json_serializer=None, mode='w'):
        self.json_serializer = json_serializer
        self.fname = fname
        self.use_tmp_file = use_tmp_file
        self.is_gz = fname.endswith('.gz')

        self.is_empty = mode == 'w' or (mode != 'w' and not fs.exists(self.fname))

        open_func = gzip.open if self.is_gz else open
        if use_tmp_file:
            if tmp_dir is not None and not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
            self.tmp_fname = tempfile.mktemp(dir=tmp_dir)
            self.tmp_stream = open_func(self.tmp_fname, mode)
        else:
            self.tmp_stream = open_func(self.fname, mode)

        self.finished = False

    def write_doc(self, doc):
        if not self.is_empty:
            self.tmp_stream.write(b'\n' if self.is_gz else '\n')

        sdoc = json.dumps(doc, default=self.json_serializer)

        if self.is_gz: sdoc = sdoc.encode('utf8')
        self.tmp_stream.write(sdoc)
        self.is_empty = False

    def flush(self):
        self.tmp_stream.flush()

    def finish(self):
        self.tmp_stream.close()
        if self.use_tmp_file:
            fs.move(self.tmp_fname, self.fname)
        self.finished = True

    def cleanup(self):
        # finish but dont write on the self.fname
        self.tmp_stream.close()
        if self.use_tmp_file:
            os.unlink(self.tmp_fname)
        self.finished = True

    def __del__(self):
        if hasattr(self, 'tmp_fname') and self.tmp_fname is not None and os.path.exists(self.tmp_fname):
            os.unlink(self.tmp_fname)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        if exc_type is None:
            self.finish()
        else:
            self.cleanup()


def write_jl(contents, fname):
    with JlWriter(fname) as writer:
        for doc in contents:
            writer.write_doc(doc)


class PathHasher:
    def __init__(self, path, n_buckets, hash_fn=None, writer_kwargs=None, hierarchical=True):
        self.path = path
        self.writers = {}
        self.hash_fn = hash_fn or mmh3.hash
        self.n_buckets = n_buckets
        self.writer_kwargs = writer_kwargs or {}
        self.hierarchical = hierarchical

    def get_writer(self, hasheable, basename):
        bucket = abs(self.hash_fn(str(hasheable))) % self.n_buckets
        if bucket not in self.writers:
            if self.hierarchical:
                folder = fs.join(self.path, str(bucket))
                fs.ensure_exists(folder)
                fname = fs.join(folder, basename)
            else:
                folder = self.path
                fs.ensure_exists(self.path)
                fname = fs.join(folder, f'{bucket}_{basename}')

            self.writers[bucket] = JlWriter(fname, **self.writer_kwargs)
        return self.writers[bucket]

    def finish(self):
        for writer in self.writers.values():
            writer.finish()

    def cleanup(self):
        for writer in self.writers.values():
            writer.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        if exc_type is None:
            self.finish()
        else:
            self.cleanup()


# class DecimalEncoder(json.JSONEncoder):
#     def default(self, o):
#         if isinstance(o, Decimal):
#             # wanted a simple yield str(o) in the next line,
#             # but that would mean a yield on the line with super(...),
#             # which wouldn't work (see my comment below), so...
#             return int(o)
#         else:
#             return super().default(o)
def write_partitioned(it, out_dir, partition_size=80_000, it_size=None):
    it = iter(it)
    fs.ensure_clean(out_dir)
    part_id = 0
    finished = False
    if it_size is not None:
        total_reminder = it_size % partition_size
        n_partitions = it_size // partition_size
        reminder_per_partition = total_reminder // n_partitions
        partition_size += reminder_per_partition
    else:
        n_partitions = None

    while not finished:
        wrote_something = False
        fname = fs.join(out_dir, f'part_{part_id:03d}.jl.gz')
        with JlWriter(fname) as writer:
            for i in range(partition_size):
                try:
                    doc = next(it)
                    writer.write_doc(doc)
                    wrote_something = True
                except StopIteration:
                    finished = True

            # Write the last few
            if n_partitions is not None and part_id == n_partitions - 1:
                for doc in it:
                    writer.write_doc(doc)
                finished = True

        if not wrote_something: fs.remove(fname)


        part_id += 1