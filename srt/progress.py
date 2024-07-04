from datetime import datetime
import os
from multiprocessing import Lock
from random import random
from time import time
import sys


class progress(object):
    """
    Log friendly progress reporter, copy of tqdm
    """
    lock = Lock()

    def __init__(self, iterable, pi=1, desc='', logger=None, tot=None, dyn_pi=None, start_position=3):
        """
        start_position: from which position of the iterable start counting time
        """
        if tot is None:
            tot = self._infer_iterable_length(iterable)

        if dyn_pi is None:
            dyn_pi = pi == 1

        self.tot = tot
        self.desc = desc
        self.initial_pi = self.pi = pi
        self.iterable = iterable
        self.pid = os.getpid()
        # el random ayuda a que en un multiprocessing no salgan todos los prints al mismo tiempo
        self.last_print = time() + (random() - 0.5) * pi
        self.logger = logger
        self.dyn_pi = dyn_pi
        self.start_position = start_position
        self.custom_msg_template = None
    
    def set_custom_msg(self, template, **params):
        self.custom_msg_template = template
        self.custom_msg_params = params

    def _infer_iterable_length(self, iterable):
        try:
            return len(iterable)
        except Exception:
            pass

        try:
            return iterable.__length_hint__()
        except Exception:
            pass

    def display_text(self, txt):
        with self.lock:
            if self.logger is None:
                print(txt)
                sys.stdout.flush()
            else:
                self.logger.info(txt)
    
    @property
    def _now(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def __iter__(self):
        if self.desc:
            if self.tot:
                desc = f' {self.desc} (total={self.tot})'
            else:
                desc = f' {self.desc}'
        elif self.tot:
            desc = f' ({self.tot})'
        else:
            desc = ''
        self.display_text('{} pid {} | starting to iterate{}, updating every {} seconds'.format(
            self._now, self.pid, desc, self.pi
        ))
        self.i = 0
        last_pi = self.pi
        if self.start_position == 0:
            self.start = time()

        empty = True
        for i, elem in enumerate(self.iterable):
            empty = False
            if i < self.start_position:
                self.start = self.now = time()
                yield elem
                continue
                
            self.i = i + 1

            now = self.now = time()
            
            pi = self.pi
            if self.dyn_pi:
                if now - self.start >= 10 * self.pi:
                    pi *= 5
                if now - self.start >= 30 * self.pi:
                    pi *= 6
                if now - self.start >= 5 * 60 * self.pi:
                    pi *= 2
                if now - self.start >= 60 * 60 * self.pi:
                    pi *= 5
                if last_pi != pi:
                    self.display_text('{} pid {} | print interval is now {}s'.format(self._now, self.pid, pi))

            if self.last_print is None or (now - self.last_print) > last_pi:
                self._print_progress()

                self.last_print = now
            
            last_pi = pi

            yield elem

        if not empty: self._print_progress()
        self.print_finish_iteration()

    def print_finish_iteration(self, msg=''):
        to_print = "{} pid {} | finished {}. Total time: {}".format(
            self._now, self.pid, self.desc, humanize_delta(time() - getattr(self, 'start', time()))
        )
        if msg:
            to_print = f'{to_print}. {msg}'

        self.display_text(to_print)

    def _print_progress(self):
        if self.tot:
            self._tot_aware_print()
        else:
            self._stream_print()

    @property
    def speed(self):
        return (self.i - self.start_position) / max(0.01, self.now - self.start)

    @property
    def elapsed(self):
        return humanize_delta(time() - self.start)

    @property
    def progress_pct(self):
        if self.tot is None: raise RuntimeError('Cannot compute progress without tot')
        return float(self.i) / self.tot * 100

    def _tot_aware_print(self):
        template = (
            "{now} pid {pid} | {i} of {tot}, {pct}%{desc}, speed: {speed:.02f} {speed_unit}, elapsed {elapsed}"
            "{eta}{custom_message}"
        )
        self.display_text(
            template.format(
                now=self._now,
                pid=self.pid,
                i=self.i,
                tot=self.tot,
                pct=f'{self.progress_pct:.01f}',
                desc=f' of {self.desc}' if self.desc else '',
                speed=self.speed if self.speed >= 1 else 1/self.speed,
                speed_unit='it/s' if self.speed >= 1 else 's/it',
                elapsed=self.elapsed,
                eta=(
                    f', eta {humanize_delta((self.tot - self.i) / self.speed)}'
                    if self.speed > 1e-5
                    else ''
                    # if self.i > 10 or time() - self.start > 300
                    # else ''
                ),
                custom_message='' if self.custom_msg_template is None else f', {self.custom_msg_template}'.format(**self.custom_msg_params)
            )
        )

    def _stream_print(self):
        self.display_text(
            "{now} pid {pid} | {i}{desc}, speed: {speed:.02f} {speed_unit}, elapsed {elapsed}{custom_message}".format(
                now=self._now,
                pid=self.pid,
                i=self.i,
                desc=f' of {self.desc}' if self.desc else '',
                speed=self.speed if self.speed >= 1 else 1/max(self.speed, 1e-10),
                speed_unit='it/s' if self.speed >= 1 else 's/it',
                elapsed=self.elapsed,
                custom_message='' if self.custom_msg_template is None else f', {self.custom_msg_template}'.format(**self.custom_msg_params)
            )
        )


def humanize_delta(delta):
    units = ['seconds', 'minutes', 'hours', 'days', 'weeks']
    factors = [60, 60, 24, 7]
    current_unit = 0
    delta = float(delta)
    while current_unit < len(factors) and delta > factors[current_unit]:
        delta /= factors[current_unit]
        current_unit += 1
    
    if abs(delta - round(delta)) < 0.02:
        template = '{} {}'
        delta = round(delta)
    else:
        template = '{:.02f} {}'
    return template.format(delta, units[current_unit])
