from functools import lru_cache, partial
from itertools import islice
from multiprocessing import Queue, Process, cpu_count
from random import Random
from time import time

import mmh3

from srt import fs  # , compute
from srt.progress import progress
from srt.serialization import compute
from srt.serialization.read import iter_fname, iter_round_robin_fnames, iter_jl_dir, chain_fnames


class SerializedIterable:
    def __init__(self, path, limit=None, tfms=None):
        self.path = path
        self.tfms = tfms or []
        if limit is not None:
            assert self.limit is None
            self.tfms.append(limit_tfm(limit))

    def __iter__(self):
        return apply_tfms(iter_fname(self.path), self.tfms)

    @property
    def limit(self):
        for t in self.tfms:
            if isinstance(t, limit_tfm): return t.n

    def copy(self):
        res = type(self)(self.path, tfms=self.tfms[:])
        return res

    def modify_limit(self, limit):
        res = self.copy()
        for i, t in enumerate(res.tfms):
            if isinstance(t, limit_tfm):
                res.tfms[i] = limit_tfm(limit)
                return res
        raise RuntimeError('called modify limit to an iterator without limit')

    def set_limit(self, limit):
        if limit is not None:
            assert self.limit is None
            return self.add_tfm(limit_tfm(limit))
        else:
            return self.no_limit()

    def replace_limit(self, limit):
        if self.limit is not None:
            return self.modify_limit(limit)
        else:
            return self.set_limit(limit)

    def no_limit(self):
        res = self.copy()
        res.tfms = [e for e in self.tfms if not isinstance(e, limit_tfm)]
        return res

    def add_projection(self, proj):
        return self.add_tfm(proj2tfm(proj))

    def filter(self, predicate):
        return self.add_tfm(pred2tfm(predicate))

    def add_tfm(self, tfm):
        res = self.copy()
        res.tfms.append(tfm)
        return res

    def add_progress(self, *, pi=1, desc='', logger=None, tot=None, dyn_pi=None, start_position=0):
        res = self.no_progress()
        if tot is None and self.tfms and isinstance(self.tfms[-1], limit_tfm):
            tot = self.tfms[-1].n

        res.tfms.append(
            progress_tfm(pi=pi, desc=desc, logger=logger, tot=tot, dyn_pi=dyn_pi, start_position=start_position)
        )
        return res

    def no_progress(self):
        res = self.copy()
        res.tfms = [e for e in self.tfms if not isinstance(e, progress_tfm)]
        return res

    @property
    def progress(self):
        t = [e for e in self.tfms if isinstance(e, progress_tfm)]
        assert len(t) <= 1
        if t: return t[0]

    @property
    def name(self):
        return fs.name(self.path)

    def __repr__(self):
        return f"{type(self).__name__}({self.path}, limit={self.limit}, tfms=[{'...' if self.tfms else ''}])"


class PartitionedIterable(SerializedIterable):
    def __init__(self, path, limit=None, tfms=None, round_robin=True):
        self.round_robin = round_robin
        super().__init__(path, limit, tfms)
        self.only_one_fname = False
        if not isinstance(path, list) and not fs.is_dir(path):
            raise ValueError(f'Path {path} must be a directory or a list of files')
        self.iter_shuffled = False
        self.iter_cores = 1

    @property
    def length(self):
        res = 0
        for pres in self.pmap(_length, results_iterator=True):
            res += pres['res']
        return res

    def take(self, n=1, seed=42):
        p = Random(seed).choice(self._sub_paths())
        pi = self.copy()
        pi.path = [p]
        return list(pi.no_limit().set_limit(n))

    def shuffle(self):
        assert self.round_robin
        pi = self.copy()
        pi.iter_shuffled = True
        return pi

    def copy(self):
        res = super().copy()
        res.round_robin = self.round_robin
        res.iter_shuffled = self.iter_shuffled
        res.iter_cores = self.iter_cores
        return res

    def sub_iters(self):
        sub_paths = self._sub_paths()
        fname_limit = None if self.limit is None else self.limit // len(sub_paths)
        residual = None if self.limit is None else self.limit % len(sub_paths)

        res = []
        # assume it is homogeneous
        sub_iter_is_dir = None
        for sub_path in sub_paths:
            if sub_iter_is_dir is None:
                sub_iter_is_dir = self._is_dir(sub_path)
            cls = PartitionedIterable if sub_iter_is_dir else SerializedIterable

            res.append(cls(sub_path, tfms=self.tfms))
            if fname_limit is not None:
                sub_iter_limit = fname_limit
                if residual:
                    sub_iter_limit += 1
                    residual -= 1
                res[-1] = res[-1].modify_limit(sub_iter_limit)

        return res

    def torch_worker_split(self):
        """
        Splits the iterable for it to be compatible with what torch multiprocessing DataLoader expects
        """
        import torch

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self

        sub_paths = self._sub_paths()

        if len(sub_paths) == worker_info.num_workers:
            sub_paths = [sub_paths[worker_info.id]]
        else:
            sub_paths = [
                p
                for p in sub_paths
                if (abs(mmh3.hash(p)) % worker_info.num_workers) == worker_info.id
            ]

        res = self.copy()
        if res.limit:
            res = res.modify_limit(
                res.limit // worker_info.num_workers +
                (worker_info.id < res.limit % worker_info.num_workers)
            )
        res.path = sub_paths
        return res

    def _sub_paths(self):
        if isinstance(self.path, list):
            return self.path
        else:
            return [p for p in fs.ls(self.path) if fs.name(p) != '_tmp']

    @lru_cache(maxsize=512)
    def _is_dir(self, p):
        return fs.is_dir(p)

    def pmap(self, func, processes=-1, maxtasksperchild=1, results_iterator=False, desc=None, print_progress=True,
             keep_input=True, **common_kwargs):
        return compute.process_file_list(
            self.sub_iters(), func, maxtasksperchild=maxtasksperchild, processes=processes,
            results_iterator=results_iterator, desc=desc, print_progress=print_progress, keep_input=keep_input,
            **common_kwargs
        )

    def chained_pmap(self, func, out_dir, processes=-1, maxtasksperchild=1, results_iterator=False, **common_kwargs):
        common_kwargs['out_dir'] = out_dir

        results = compute.process_file_list(
            self.sub_iters(), func, maxtasksperchild=maxtasksperchild, processes=processes,
            results_iterator=results_iterator, **common_kwargs
        )
        return IntermediateResults(results, out_dir)

    def ensure_round_robin(self):
        res = self.copy()
        res.round_robin = True
        return res

    def ensure_chained(self):
        assert not self.iter_shuffled
        res = self.copy()
        res.round_robin = False
        return res

    def set_to_test(self):
        """
        iterates only one fname. Useful for testing
        """
        res = self.copy()
        res.only_one_fname = True
        return res

    def multicore(self, cores=-1):
        """
        Enables multicore iteration. If cores=-1, uses all available cores
        Consider that, if enabled the order of the results is not guaranteed
        """
        res = self.copy()
        if cores == -1: cores = cpu_count()
        res.iter_cores = cores
        return res

    def __iter__(self):
        if self.iter_cores == 1:
            yield from self._single_core_iter()
        else:
            it = self._multicore_iter()
            if self.progress:
                it = self.progress(it)
            yield from it

    def _single_core_iter(self):
        if self.only_one_fname:
            if isinstance(self.path, list):
                fname = self.path[0]
            else:
                fname = fs.ls(self.path)[0]
            return apply_tfms(iter_fname(fname), self.tfms)
        else:
            if self.round_robin:
                it_func = iter_round_robin_fnames if isinstance(self.path, list) else iter_jl_dir
                if self.iter_shuffled:
                    it_func = partial(it_func, seed=time())
            else:
                it_func = (
                    chain_fnames
                    if isinstance(self.path, list)
                    else partial(iter_jl_dir, round_robin=False, shuffle_fnames=False)
                )

            return apply_tfms(it_func(self.path), self.tfms)

    def _multicore_iter(self):
        # Dataloader style read
        output_queue = Queue()
        pool = []
        inputs = [e.no_progress() for e in self.sub_iters()]

        while len(inputs) + len(pool) > 0:
            pool = [e for e in pool if e.is_alive()]
            while len(pool) < self.iter_cores and len(inputs) > 0:
                it = inputs.pop()
                proc = Process(target=write, args=(it, output_queue))
                proc.start()
                pool.append(proc)

            if not output_queue.empty():
                yield output_queue.get()

        while not output_queue.empty():
            yield output_queue.get()


def write(it, queue):
    for doc in it:
        queue.put(doc)


class InterleavedPartitionedIterable(PartitionedIterable):
    def __init__(self, path, limit=None, tfms=None, round_robin=True, restart_iters=True):
        super().__init__(path, limit, tfms, round_robin)
        sub_paths = fs.ls(path)
        self.restart_iters = restart_iters
        self.piters = [
            PartitionedIterable(p, round_robin=round_robin)
            for p in sub_paths
        ]
        self.pi_weight_dict = None

    def set_sub_pi_weight(self, weight_dict):
        for p in self.piters:
            assert p.name in weight_dict

        res = self.copy()
        res.pi_weight_dict = weight_dict
        return res

    def take(self, n=1, seed=42):
        p = Random(seed).choice(self._sub_paths())
        pi = PartitionedIterable(p, tfms=self.tfms)
        return list(pi.no_limit().set_limit(n))

    def copy(self):
        res = super().copy()
        res.pi_weight_dict = self.pi_weight_dict
        res.restart_iters = self.restart_iters
        res.piters = [e.copy() for e in self.piters]
        return res

    def _get_piters_w_tfms(self):
        res = []
        if self.pi_weight_dict is None:
            limit = None if self.limit is None else self.limit // len(self.piters)
            limits = [limit for pi in self.piters]
        else:
            limits = []
            s = sum(self.pi_weight_dict.values())
            for pi in self.piters:
                limits.append(self.limit * self.pi_weight_dict[pi.name] // s)

        residual = None if self.limit is None else self.limit - sum(limits)
        while residual is not None and residual > len(self.piters):
            for i in range(len(limits)):
                limits[i] += 1
            residual -= len(limits)

        for i, p in enumerate(self.piters):
            for t in self.tfms:
                # TODO: should we copy this? for pmap wouldnt make sense...
                if isinstance(t, progress_tfm): continue
                if isinstance(t, limit_tfm):
                    limit = limits[i]
                    if residual:
                        limit += 1
                        residual -= 1
                    t = limit_tfm(limit)
                p = p.add_tfm(t)
            res.append(p)
        return res

    def sub_iters(self):
        res = []
        for p in self._get_piters_w_tfms():
            res.extend(p.sub_iters())
        return res

    def torch_worker_split(self):
        res = self.copy()
        res.piters = [e.torch_worker_split() for e in res.piters]
        return res

    def ensure_round_robin(self):
        res = self.copy()
        res.piters = [p.ensure_round_robin() for p in res.piters]
        return res

    def ensure_chained(self):
        res = self.copy()
        res.piters = [p.ensure_chained() for p in res.piters]
        return res

    def set_to_test(self):
        res = self.copy()
        res.piters = [p.set_to_test() for p in res.piters]
        return res

    def shuffle(self):
        res = self.copy()
        res.piters = [p.shuffle() for p in res.piters]
        return res

    def __iter__(self):
        res = self._iterator()
        if self.progress:
            res = self.progress(res)
        return res

    def _iterator(self):
        piters = self._get_piters_w_tfms()
        its = [iter(p) for p in piters]
        finished_iters = set()
        while len(finished_iters) < len(its):
            for i, it in enumerate(its):

                n = 1 if self.pi_weight_dict is None else self.pi_weight_dict[piters[i].name]
                to_yield = list(islice(it, n))
                for e in to_yield: yield e

                if len(to_yield) < n:
                    if self.restart_iters:
                        its[i] = iter(piters[i])
                    finished_iters.add(i)
                if len(finished_iters) == len(its): break


class IntermediateResults(PartitionedIterable):
    def __init__(self, results, path, limit=None, tfms=None):
        super().__init__(path, limit, tfms)
        self.results = results

    def copy(self):
        return IntermediateResults(self.results, self.path, self.limit, self.tfms)


def apply_tfms(it, tfms):
    for tfm in tfms:
        it = tfm(it)
    yield from it


class limit_tfm:
    def __init__(self, n):
        self.n = n

    def __call__(self, it):
        yield from islice(it, self.n)

    def __repr__(self):
        return f'limit({self.n})'


class progress_tfm:
    def __init__(self, pi, desc, logger, tot, dyn_pi, start_position):
        self.pi = pi
        self.desc = desc
        self.logger = logger
        self.tot = tot
        self.dyn_pi = dyn_pi
        self.start_position = start_position

    def __call__(self, it):
        yield from progress(it, pi=self.pi, desc=self.desc, logger=self.logger,
                            tot=self.tot, dyn_pi=self.dyn_pi,
                            start_position=self.start_position)


class proj2tfm:
    def __init__(self, proj):
        self.proj = proj

    def __call__(self, it):
        proj = self.proj
        for doc in it:
            yield proj(doc)


class pred2tfm:
    def __init__(self, pred):
        self.pred = pred

    def __call__(self, it):
        pred = self.pred
        for doc in it:
            if pred(doc): yield doc


def _length(it):
    n = 0
    for _ in it: n += 1
    return n
