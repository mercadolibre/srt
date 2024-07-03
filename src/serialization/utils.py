from collections import Counter
from functools import wraps

import urllib3

from src.progress import progress


class boto_retry:
    """
    Sometimes there's a connection error when you iterate a file from S3.
    One kinda ugly way of solving it is by making your function idempotent and
    having a decorator that retries running the function
    """

    def __init__(self, times=5):
        self.times = times

    def __call__(self, func):
        return self.decorate_func(func)

    def decorate_func(self, func):
        @wraps(func)
        def new_f(*args, **kwargs):
            for _ in range(self.times):
                try:
                    return func(*args, **kwargs)
                except urllib3.exceptions.ProtocolError as e:
                    with progress.lock:
                        print(f'func {func.__name__} failed because of a boto3 protocol error. Retrying...')
            raise e

        return new_f


def length(pi):
    res = 0
    for pres in pi.pmap(_length):
        res += pres['res']
    return res


def _length(it):
    res = 0
    for _ in it: res += 1
    return res


def simple_value_counts(it, *keys):
    res = Counter()
    for pres in it.pmap(_simple_vc, keys=keys):
        res.update(pres['res'])
    return res


def _simple_vc(it, keys):
    res = Counter()
    keys = [key.split('.') for key in keys]
    for doc in it:
        vals = []
        for key in keys:
            val = None
            v = doc
            for k in key:
                v = v.get(k)
                if v is None: break
            else:
                val = v
            vals.append(val)

        if len(keys) == 1:
            res[vals[0]] += 1
        else:
            res[tuple(vals)] += 1
    return res
