import json
import logging
import pickle as pkl

from srt.conf_utils import ConfigManager
from srt.fine_tune.elastic_weight_consolidation import EWCPenalty

logging.basicConfig(level=logging.INFO)

import sys
from logging import getLogger

import torch

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_environment,
)
from srt import fs, settings
from srt.settings import RECBOLE_DIR

here = fs.parent(fs.abspath(__file__))
properties_path = fs.join(RECBOLE_DIR, 'recbole/properties/model/')

config_file_list = [fs.join(properties_path, 'SASRecF2' + '.yaml')]


def fit_ewc(config_dict_fname, input_checkpoint_fname, large_dt_name, ewc_lambda, batch_size, output_fname):
    config_dict = load_config_dict(config_dict_fname)
    config_dict['train_batch_size'] = batch_size
    config = Config(
        model=config_dict['model'],
        dataset=large_dt_name,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )

    # dataset filtering
    large_dataset = create_dataset(config)
    large_train_data, large_valid_data, large_test_data = data_preparation(config, large_dataset)
    model = get_model(config["model"])(config, large_train_data._dataset).to(config["device"])

    checkpoint = torch.load(input_checkpoint_fname, map_location=config["device"])

    for k in list(checkpoint["state_dict"].keys()):
        if 'total_ops' in k or 'total_params' in k:
            checkpoint["state_dict"].pop(k)

    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    EWCPenalty(ewc_lambda).fit(model, large_train_data, n_batches=3000).save(output_fname)


def adapt_model(config_dict_fname, input_checkpoint_fname, run_name, exp_name, large_dt_name, small_dt_name,
                output_checkpoint_fname, output_dataset_fname):
    config_dict = load_config_dict(config_dict_fname)

    config_dict['mlflow']['run_name'] = run_name
    config_dict['mlflow']['experiment_name'] = exp_name

    config, dataset, train_data, valid_data, test_data = get_large_dt(config_dict, large_dt_name)

    small_dt, tr_beauty, val_beauty, te_beauty = load_other_dt(small_dt_name, config_dict, dataset)

    small_dt.field2id_token['title'] = dataset.field2id_token['title']
    small_dt.field2id_token['brand'] = dataset.field2id_token['brand']

    model = load_model_remap(config, dataset, small_dt, checkpoint_fname=input_checkpoint_fname)

    print('item features')
    print(model.feature_embed_layer.item_feat)

    print('example')
    print(small_dt.field2id_token['title'][model.feature_embed_layer.item_feat[1].title.cpu().numpy()])

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)  # .to('cpu'))
    eval_result = trainer.evaluate(
        te_beauty,
        load_best_model=False, show_progress=True,
        neg_sample_func=train_data._neg_sampling, phase='test'
    )
    print(eval_result)

    trainer.config.variable_config_dict['remap_dataset'] = None
    model.feature_embed_layer.dataset = None

    print(f'Saving adapted model to {output_checkpoint_fname}')
    trainer._save_checkpoint(epoch=0, saved_model_file=output_checkpoint_fname)

    small_dt.config['remap_dataset'] = None
    small_dt.config.external_config_dict['remap_dataset'] = None

    print(f'Saving adapted dataset to {output_dataset_fname}')
    with open(output_dataset_fname, 'wb') as f:
        pkl.dump(small_dt, f, pkl.HIGHEST_PROTOCOL)

    return eval_result


def load_config_dict(config_dict_fname):
    with open(config_dict_fname, 'r') as f:
        config_dict = json.load(f)
    config_dict['eval_args']['mode']['train'] = 'full'
    config_dict['eval_args']['mode']['test'] = 'full'
    config_dict['eval_args']['mode']['valid'] = 'full'
    config_dict['data_path'] = settings.RECBOLE_DATASETS
    return config_dict


def from_scratch(*, small_dt_name, epochs, out_fname, train_batch_size, learning_rate, hidden_size,
                 n_layers, run_name, exp_name):
    config_dict = get_from_scratch_config_dict(epochs, exp_name, hidden_size, learning_rate, n_layers, run_name,
                                               small_dt_name, train_batch_size)
    run_experiment('SASRecF2', small_dt_name, config_dict, out_fname)
    return config_dict


def get_from_scratch_config_dict(epochs, exp_name, hidden_size, learning_rate, n_layers, run_name, small_dt_name,
                                 train_batch_size):
    config_dict = get_config_dict(
        small_dt_name, 'SASRecF2', epochs, evaluation_mode='full', experiment_name=exp_name, run_name=run_name,
        overrides=dict(
            train_batch_size=train_batch_size, learning_rate=learning_rate,
            hidden_size=hidden_size, inner_size=hidden_size, n_layers=n_layers,
            nce_num_negatives=300, nce_sampling_strategy='popularity',
            nce_global_negatives=False, scheduler={'type': 'one-cycle'}, stopping_step=5,
            nce_temperature=1, nce_label_smoothing=0.0,
            eval_batch_size=256
        )
    )
    config_dict['data_path'] = settings.RECBOLE_DATASETS
    return config_dict


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

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    # XXX hack
    trainer.saved_model_file = out_fname

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"], verbose=True
    )

    # model evaluation
    trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"],
        # TODO: remove this!
        neg_sample_func=train_data._neg_sampling, phase='test'
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    # logger.info(set_color("test result", "yellow") + f": {test_result}")

    return trainer


def fine_tune(*, config_dict_fname, small_dt_fname, input_checkpoint_fname,
              epochs, freeze_embeddings, unfreeze_n_layers, output_checkpoint_fname,
              run_name, exp_name, train_batch_size=None, learning_rate=None, ewc_fname=None):
    config_dict = load_config_dict(config_dict_fname)

    config_dict['save_dataset'] = False
    config_dict['mlflow']['run_name'] = run_name
    config_dict['mlflow']['experiment_name'] = exp_name
    config_dict['mlflow']['auto_continue'] = True

    if train_batch_size:
        config_dict['train_batch_size'] = train_batch_size

    if learning_rate:
        config_dict['learning_rate'] = learning_rate

    with open(small_dt_fname, 'rb') as f:
        small_dt = pkl.load(f)

    config_dict['gradient_accumulation_steps'] = 10
    config = Config(
        model='SASRecF2',
        dataset=fs.strip_ext(fs.name(small_dt_fname)),
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    print(config)
    init_seed(config["seed"], config["reproducibility"])

    tr_beauty, val_beauty, te_beauty = data_preparation(config, small_dt)
    model, checkpoint = load_model(config, small_dt, input_checkpoint_fname)

    print('item features')
    print(model.feature_embed_layer.item_feat)

    print('example')
    print(small_dt.field2id_token['title'][model.feature_embed_layer.item_feat[1].title.cpu().numpy()])

    # model.sampling_strategy = 'knn'
    # model.nce_knn_random_frac = 0.5
    # model.nce_knn_hard_negatives_pool_size = 300
    # model.nce_knn_discard_neighbors = 100
    # model.nce_knn_tot_neighbors = 400
    # model.on_epoch_start(increase_epoch=False)

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.metrics_logger.log_params(dict(freeze_embeddings=freeze_embeddings, unfreeze_n_layers=unfreeze_n_layers, ))
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])

    print('Evaluation sanity')
    print("*" * 40)
    print(trainer.evaluate(
        te_beauty, load_best_model=False, show_progress=True, neg_sample_func=tr_beauty._neg_sampling, phase='test')
    )
    print("*" * 40)

    config['epochs'] = epochs
    config['eval_step'] = 1
    freeze_model(model, freeze_embeddings=freeze_embeddings, n_layers=unfreeze_n_layers)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    if ewc_fname is not None:
        trainer.loss_callback = EWCPenalty.load(ewc_fname).get_penalty

    trainer.fit(tr_beauty, val_beauty, show_progress=True, callback_fn=None, saved=False)

    print(f'Saving checkpoint on {output_checkpoint_fname}')
    trainer._save_checkpoint(epoch=0, saved_model_file=output_checkpoint_fname)

    print(trainer.evaluate(
        te_beauty,
        load_best_model=False, show_progress=True,
        neg_sample_func=tr_beauty._neg_sampling, phase='test'
    ))


def get_fine_tune_checkpoints_dir(task_id, run_name=None):
    out_dir = fs.join('fine_tuning', task_id)
    if run_name:
        out_dir = fs.join(out_dir, run_name)
    chkp_dir = fs.ensure_exists(fs.join(out_dir, 'checkpoints'))
    return chkp_dir


def get_adapted_dt_fname(task_id, small_dt_name):
    out_dir = fs.ensure_exists(fs.join('fine_tuning', task_id))
    dt_dir = fs.ensure_exists(fs.join(out_dir, 'datasets'))
    return fs.join(dt_dir, f'{small_dt_name}.pkl')


def get_large_dt(config_dict, dataset):
    model = 'SASRecF2'
    # dataset = 'pruned_amazon-1M'

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
    # logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    print('Large dataset')
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    return config, dataset, train_data, valid_data, test_data


def load_model(config, dataset, checkpoint_fname=None, device=None):
    device = device or config['device']
    config['device'] = device

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, dataset).to(config["device"])

    if checkpoint_fname:
        checkpoint = torch.load(checkpoint_fname, map_location=config['device'])

        for k in list(checkpoint["state_dict"].keys()):
            if 'total_ops' in k or 'total_params' in k:
                checkpoint["state_dict"].pop(k)

        # item_feat = model.feature_embed_layer.item_feat

        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))

        # model.feature_embed_layer.item_feat = item_feat
        return model, checkpoint
    else:
        return model


def evaluate(model, data, config):
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    model.eval()
    return trainer.evaluate(
        data, load_best_model=False, show_progress=config["show_progress"],
        # TODO: remove this!
        neg_sample_func=data._neg_sampling, phase='test'
    )


def load_other_dt(dataset, config_dict, pruned_dt=None):
    if pruned_dt is not None:
        config_dict['remap_dataset'] = pruned_dt
    config_dict['train_batch_size'] = 200
    config_dict['save_dataset'] = False

    config = Config(
        model='SASRecF2',
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])

    init_logger(config)
    logger = getLogger()

    dataset = create_dataset(config)

    print('Small dataset')
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    return dataset, train_data, valid_data, test_data


def freeze_model(model, freeze_embeddings, n_layers):
    model.requires_grad_(True)

    if freeze_embeddings:
        print('Freezing embeddings')
        model.feature_embed_layer.requires_grad_(False)
        model.position_embedding.requires_grad_(False)

    if n_layers == -1: n_layers = len(model.trm_encoder.layer)

    for i, layer in enumerate(model.trm_encoder.layer):
        if i < len(model.trm_encoder.layer) - n_layers:
            print('Freezing layer', i)
            layer.requires_grad_(False)


def load_model_remap(config, large_dt, small_dt, checkpoint_fname=None):
    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, large_dt).to(config["device"])

    small_model = get_model(config["model"])(config, small_dt)

    if checkpoint_fname:
        checkpoint = torch.load(checkpoint_fname, map_location=config["device"])

        for k in list(checkpoint["state_dict"].keys()):
            if 'total_ops' in k or 'total_params' in k:
                checkpoint["state_dict"].pop(k)

        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))

        model.feature_embed_layer.item_feat = small_model.feature_embed_layer.item_feat.to(config["device"])
        model.feature_embed_layer.item_feat.title[
            model.feature_embed_layer.item_feat.title >= len(large_dt.field2id_token['title'])] = 0
        model.feature_embed_layer.item_feat.brand[
            model.feature_embed_layer.item_feat.brand >= len(large_dt.field2id_token['brand'])] = 0

        model.update_item_distr(small_dt)

    return model


class progressive_unfreeze_callback:
    def __init__(self, model, from_scratch, fine_tuning_strategy):
        self.model = model
        self.from_scratch = from_scratch
        self.fine_tuning_strategy = fine_tuning_strategy

        if from_scratch or fine_tuning_strategy == 'full':
            self.params = {}
        else:
            freeze_embeddings = fine_tuning_strategy == 'progressive-embs-last'
            freeze_model(model, freeze_embeddings, n_layers=1)

            self.params = {
                # epoch: {'freeze_embeddings': True/False, 'n_layers': int}
                # epoch is the number of the last finished epoch (starting from 0)
                2: {'freeze_embeddings': freeze_embeddings, 'n_layers': 2},
                4: {'freeze_embeddings': freeze_embeddings, 'n_layers': 4},
                6: {'freeze_embeddings': freeze_embeddings, 'n_layers': 8},
                8: {'freeze_embeddings': freeze_embeddings, 'n_layers': -1},
                10: {'freeze_embeddings': False, 'n_layers': -1},
            }

    def __call__(self, epoch_idx, val_score):
        if epoch_idx in self.params:
            print(f'Changing freezing of model for epoch {epoch_idx + 1}')
            freeze_model(self.model, **self.params[epoch_idx])


default_overrides = dict(
    epochs=4,
    train_batch_size=512, eval_batch_size=512,
    scheduler={'type': 'one-cycle'}, stopping_step=20, eval_step=2,
    nce_temperature=1,
    nce_label_smoothing=0.0,
    nce_global_negatives=True,
    nce_sampling_strategy='popularity',
    nce_num_negatives=1000,
    save_dataset=True,
    inner_size=64, hidden_size=64,

)


def get_config_dict(dataset, model='SASRecF2', epochs=50, loss_type='InfoNCE-quick', overrides=None,
                    evaluation_mode='sampled10000', experiment_name='exp', run_name='run'):
    overrides = overrides or default_overrides
    return (
        ConfigManager(model=model, dataset=dataset)
        .specify_optimization(train_batch_size=512 * 4, eval_batch_size=512 * 4, stopping_step=20,
                              loss_type=loss_type, epochs=epochs)
        .specify_architecture(hidden_dropout_prob=0.5, attn_dropout_prob=0.5, n_layers=2,
                              hidden_size=64, inner_size=64,
                              max_item_list_length=50)
        .add_evaluation_args(mode={'valid': evaluation_mode, 'test': evaluation_mode})
        .override(**overrides)
        .mlfow(tracking_uri='sqlite:///mlruns.sqlite', experiment_name=experiment_name, run_name=run_name)
        .no_negative_sampling()
    ).config_dict
