import gc
from threading import Thread
from time import sleep

import torch
from pynvml.smi import nvidia_smi

from recbole.data.dataloader.general_dataloader import TrainDataLoader
from srt import fs
from srt.settings import RECBOLE_DIR

here = fs.parent(fs.abspath(__file__))
properties_path = fs.join(RECBOLE_DIR, 'recbole/properties/model/')


def get_free_mem():
    nvsmi = nvidia_smi.getInstance()
    return nvsmi.DeviceQuery('memory.free')['gpu'][0]['fb_memory_usage']['free']


class MemoryWorker(Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.free_mem = get_free_mem()
        self.should_run = True

    def run(self):
        while self.should_run:
            self.free_mem = min(self.free_mem, get_free_mem())
            sleep(0.1)


def gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def update_batch_size(dataloader, phase, batch_size):
    dataloader.config[f'{phase}_batch_size'] = batch_size
    return type(dataloader)(dataloader.config, dataloader.dataset, dataloader.sampler)


def estimate_mem_for_bs(trainer, train_data, batch_size):
    gc_cuda()
    w = MemoryWorker()
    w.start()
    try:
        trainer._train_epoch(
            update_batch_size(train_data, 'train', batch_size), 0, show_progress=False, total_batches=50
        )
    except torch.cuda.OutOfMemoryError:
        return None
    w.should_run = False
    w.join()
    gc_cuda()
    return w.free_mem


def get_max_batch_size(trainer, train_data, target_free=1):
    x = [32, 64, 128, 192]
    y = [estimate_mem_for_bs(trainer, train_data, bs) for bs in x]

    # Discard the out of memory errors
    to_keep = [i for i, e in enumerate(y) if e is not None]
    x = [x[i] for i in to_keep]
    y = [y[i] for i in to_keep]
    
    step = min([(y[i] - y[j])/(x[i] - x[j]) for i in range(len(x)) for j in range(len(x)) if i != j])
    
    batch_size = int(x[1] + (target_free * 1024 - y[1]) / step)

    print(f'Optimal batch size: {batch_size} (free mem: {estimate_mem_for_bs(trainer, train_data, batch_size)})')

    return batch_size


