import argparse
import json
import logging

from thop import profile
from torch import nn

logging.basicConfig(level=logging.INFO)

import sys
from logging import getLogger

import torch

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_environment,
)
from srt import fs, settings
from srt.estimate_batch_size import get_max_batch_size, update_batch_size, gc_cuda
from srt.settings import RECBOLE_DIR

here = fs.parent(fs.abspath(__file__))
properties_path = fs.join(RECBOLE_DIR, 'recbole/properties/model/')


def main(config_dict_fname, out_fname):
    with open(config_dict_fname) as f:
        config_dict = json.load(f)

    # make sure recbole finds our datasets
    config_dict['data_path'] = settings.RECBOLE_DATASETS
    run_experiment(config_dict['model'], config_dict['dataset'], config_dict, out_fname)


class M(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, b):
        return self.model.calculate_loss(b)


def compute_flops(batch, model):
    batch = batch.to('cuda')
    # TODO: uniformize ids with meli
    # increase_epoch=False to avoid recompute of hard negatives when training actually starts
    model.on_epoch_start(increase_epoch=False)

    forward_flops, params = profile(model, inputs=(getattr(batch, model.ITEM_SEQ), batch.item_length))

    return dict(
        forward_flops=forward_flops, thop_params=params,
        total_flops=profile(M(model), inputs=(batch,))[0]
    )


def run_experiment(model, dataset, config_dict, out_fname):
    config_file_list = [fs.join(properties_path, model + '.yaml')]
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)
    if config['estimate_batch_size']:
        # disable mlflow to find the batch size
        config['mlflow']['enable'] = False
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        batch_size = get_max_batch_size(trainer, train_data)
        train_data = update_batch_size(train_data, 'train', batch_size)
        valid_data = update_batch_size(valid_data, 'eval', batch_size)
        test_data = update_batch_size(test_data, 'eval', batch_size)
        config['learning_rate'] *= batch_size / config['train_batch_size']

        # create new trainer and model to start from scratch
        del trainer, model
        gc_cuda()

        # re instance everything and enable mlflow
        model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
        config['mlflow']['enable'] = True

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.saved_model_file = out_fname

    transform = construct_transform(config)

    batch = next(iter(train_data))
    flops_dict = compute_flops(batch, model)
    for k, v in flops_dict.items():
        logger.info(set_color(k, "blue") + f": {v}")
    trainer.metrics_logger.log_params(flops_dict)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"],
    )

    # model evaluation
    try:
        trainer.evaluate(
            test_data, load_best_model=True, show_progress=config["show_progress"],
            # TODO: remove this!
            neg_sample_func=train_data._neg_sampling, phase='test'
        )
    except torch.cuda.OutOfMemoryError:
        logger.info(f'CUDA Memory Error when evaluating with test set')
        pass

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    # logger.info(set_color("test result", "yellow") + f": {test_result}")

    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-dict', help='config dict json file', required=True, default='config_dict.json')
    parser.add_argument('-o', '--out-fname', help='output checkpoint fname', required=True)

    args = parser.parse_args()

    main(args.config_dict, args.out_fname)
