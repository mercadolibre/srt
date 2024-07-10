import os
from contextlib import contextmanager
from datetime import datetime
from time import time

from srt.progress import humanize_delta


class TimeObject(object):
    def __init__(self, name):
        self.name = name
        self.value = None


@contextmanager
def timeit(txt=None, silent=False):
    txt = txt or 'measure time'
    if not silent:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} pid {os.getpid()} | Starting to {txt}')

    t0 = time()
    to = TimeObject(txt)
    yield to
    to.value = time() - t0

    if not silent:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} pid {os.getpid()} | Time for {txt} is '
              f'{humanize_delta(to.value)}')
