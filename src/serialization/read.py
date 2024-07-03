import json
import csv
from concurrent.futures import ThreadPoolExecutor
from random import Random

from src import fs


class iter_jl:
    def __init__(self, fname, compressed=None, limit=None, prefetch=False):
        #         if not fs.exists(fname): raise RuntimeError(f'Cannot iterate {fname}. Doesnt exist')
        self.fname = fname
        self.compressed = compressed
        self.limit = limit
        self.prefetch = prefetch

        self.stream = None
        if prefetch: self._setup_iter()
        self._it = None

    @property
    def name(self):
        return fs.name(self.fname)

    def __next__(self):
        if self._it is None:
            self._it = self._iterator()
        return next(self._it)

    def __iter__(self):
        if self._it is None:
            self._it = self._iterator()
        return self._it

    def _setup_iter(self):
        self.stream = fs.smart_open(self.fname, 'rb', compressed=self.compressed, prefetch=self.prefetch)

    def _iterator(self):
        if self.stream is None: self._setup_iter()
        limit = self.limit
        try:
            for i, line in enumerate(self.stream):
                if limit is not None and limit == i: break
                try:
                    yield json.loads(line)
                except json.decoder.JSONDecodeError:
                    print(f'Error decoding line {i} in {self.fname}')
        finally:
            self.stream.close()


class iter_csv(iter_jl):
    def __init__(self, fname, compressed=None, limit=None, prefetch=False, fieldnames=None):
        super().__init__(fname, compressed, limit, prefetch)
        self.fieldnames = fieldnames

    def _setup_iter(self):
        self.stream = fs.smart_open(self.fname, 'r', compressed=self.compressed, prefetch=self.prefetch)

    def _iterator(self):
        if self.stream is None: self._setup_iter()
        limit = self.limit
        reader = csv.DictReader(self.stream, fieldnames=self.fieldnames)
        try:
            for i, doc in enumerate(reader):
                if limit is not None and limit == i: break
                yield doc
        finally:
            self.stream.close()


def is_csv(fname):
    return fname.endswith('.csv') or fname.endswith('.csv.gz')


def is_jsonlines(fname):
    return fname.endswith('.jl') or fname.endswith('.jl.gz')


def iter_fname(fname, *args, **kwargs):
    if is_csv(fname):
        return iter_csv(fname, *args, **kwargs)
    elif is_jsonlines(fname):
        return iter_jl(fname, *args, **kwargs)
    else:
        raise RuntimeError('invalid fname')


def iter_jl_dir(path, round_robin=True, limit=None, shuffle_fnames=True, seed=42):
    # TODO make sure there are only <=1 check of these through the flow
    if not fs.exists(path): raise RuntimeError(f'Cannot iterate {path}. Doesnt exist')
    if round_robin:
        for e in iter_round_robin_dir(path, limit=limit, shuffle_fnames=shuffle_fnames, seed=seed):
            yield e
    else:
        if shuffle_fnames: raise ValueError('Doesnt make any sense. Set round_robin to True')
        fnames = sorted([p for p in fs.ls(path) if fs.name(p) != '_tmp'])  # TODO: check this
        yield from chain_fnames(fnames, limit)


def chain_fnames(fnames, limit=None, keep_order=False):
    fname_limit = None if limit is None else limit // len(fnames)
    residual = None if limit is None else limit % len(fnames)
    if not keep_order: fnames.sort()
    # TODO: simplify this, no need to pass 2 parameters
    its = _get_warmed_up_iterators(fnames, fname_limit=fname_limit, residual=residual, sort_fnames=not keep_order)
    for it in its:
        for doc in it:
            yield doc


def iter_round_robin_dir(path, limit=None, shuffle_fnames=True, seed=42):
    if not fs.exists(path): raise RuntimeError(f'Cannot iterate {path}. Doesnt exist')
    fnames = [p for p in fs.ls(path) if fs.name(p) != '_tmp']  # TODO: check this
    fnames.sort()
    return iter_round_robin_fnames(fnames, limit=limit, shuffle_fnames=shuffle_fnames, seed=seed)


def iter_round_robin_fnames(fnames, limit=None, shuffle_fnames=True, seed=42):
    fname_limit = None if limit is None else limit // len(fnames)
    residual = None if limit is None else limit % len(fnames)
    fnames.sort()
    its = _get_warmed_up_iterators(fnames, fname_limit=fname_limit, residual=residual)

    rnd = Random(seed)
    while len(its) > 0 and (limit is None or limit > 0):
        to_remove = []
        if shuffle_fnames: rnd.shuffle(its)
        for i, it in enumerate(its):
            times = rnd.randint(1, 15 if shuffle_fnames else 1)
            try:
                for _ in range(times):
                    yield next(it)
                    if limit is not None:
                        limit -= 1
                        if limit == 0: break
            except StopIteration:
                to_remove.append(i)

            if limit == 0: break

        if to_remove:
            to_remove = set(to_remove)
            its = [it for i, it in enumerate(its) if i not in to_remove]


def _get_warmed_up_iterators(fnames, fname_limit=None, residual=None, sort_fnames=True):
    with ThreadPoolExecutor(max_workers=40) as e:
        its = []
        if sort_fnames: fnames = sorted(fnames)
        for fname in fnames:  # sort fnames to make sure residual is always distributed the same way
            this_limit = fname_limit
            if residual:
                this_limit += 1
                residual -= 1
            its.append(e.submit(iter_fname, fname=fname, limit=this_limit, prefetch=True))
        its = [f.result() for f in its]
    return its
