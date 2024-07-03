import argparse
import json
import uuid
from multiprocessing import Process

from mlflow import MlflowClient

from fine_tune_utils import adapt_model, fine_tune as fine_tune_model, from_scratch as train_model_from_scratch, \
    get_from_scratch_config_dict, fit_ewc
from recbole.utils.mlflow_logger import get_or_create_experiment
from src import fs

here = fs.abspath(fs.parent(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--fine-tune-config-dict', help='fine tune config dict json file',
                        required=True, default='fine_tune_config_dict.json')

    parser.add_argument('-s', '--small-dt-name', help='dataset name to train on',
                        required=True, default='amazon-beauty')

    parser.add_argument('-l', '--large-dt-name', help='dataset used to train the model to be fine-tuned',
                        required=False)

    parser.add_argument('-i', '--initial-checkpoint', help='checkpoint to start fine-tuning from',
                        required=False)

    parser.add_argument('-p', '--pre-train-config-dict', help='config dict file used to obtain the initial checkpoint',
                        required=False)

    args = parser.parse_args()
    with open(args.fine_tune_config_dict) as f:
        fine_tune_config = json.load(f)

    if fine_tune_config['method'] != 'from_scratch':
        if args.large_dt_name is None or args.initial_checkpoint is None or args.pre_train_config_dict is None:
            raise ValueError('Need to provide large_dt_name, initial_checkpoint and pre_train_config_dict')

    main_internal(args.small_dt_name, fine_tune_config, args.large_dt_name, args.pre_train_config_dict, args.initial_checkpoint)


def main_internal(small_dt_name, fine_tune_config, large_dt_name,
                  pretrain_config_dict_fname=None, starting_checkpoint_fname=None):
    """
    pretrain_config_dict_fname is the config_dict used to obtain the starting_checkpoint_fname
    """
    keys_by_method = {
        'progressive': ['epochs_per_layer', 'final_epochs', 'n_layers'],  # ewc_lambda optional
        'full': ['epochs', 'freeze_embeddings', 'unfreeze_n_layers'],
        'from_scratch': ['epochs', 'n_layers', 'hidden_size'],
        'only_embeddings': ['final_epochs'],
        'ewc': ['epochs', 'ewc_lambda', 'freeze_embeddings', 'unfreeze_n_layers'],
    }

    common_keys = [
        'method', 'learning_rate', 'train_batch_size', 'mlflow'
    ]

    if fine_tune_config['method'] == 'progressive':
        with open(pretrain_config_dict_fname) as f:
            fine_tune_config['n_layers'] = json.load(f)['n_layers']

    for k in common_keys: assert k in fine_tune_config, k
    for k in keys_by_method[fine_tune_config['method']]: assert k in fine_tune_config, k

    fine_tune_config['large_dt_name'] = large_dt_name
    fine_tune_config['input_checkpoint_fname'] = starting_checkpoint_fname
    fine_tune_config['small_dt_name'] = small_dt_name

    if fine_tune_config['method'] != 'from_scratch':
        fine_tune_config['config_dict_fname'] = pretrain_config_dict_fname

    run_name = fine_tune_config['run_name'] = str(uuid.uuid4())
    working_dir = fine_tune_config['working_dir'] = fs.ensure_exists(fs.abspath(fs.join('working_dir', run_name)))
    fine_tune_config['out_fname'] = fs.join(working_dir, 'final.pth')

    fine_tune_config = run(fine_tune_config) or fine_tune_config

    # log extra params of this run
    extra_params = dict(method=fine_tune_config['method'])
    if fine_tune_config['method'] == 'progressive':
        extra_params.update(dict(
            pochs_per_layer=fine_tune_config['epochs_per_layer'],
            final_epochs=fine_tune_config['final_epochs']
        ))
    elif fine_tune_config['method'] == 'only_embeddings':
        extra_params.update(dict(
            final_epochs=fine_tune_config['final_epochs']
        ))

    # Fine-tune config does not determine mlflow configuration to avoid redundancies
    with open(pretrain_config_dict_fname) as f:
        pretrain_config_dict = json.load(f)
    mlflow_client = MlflowClient(pretrain_config_dict['mlflow']['tracking_uri'])

    exp_id = mlflow_client.get_experiment_by_name(fine_tune_config['mlflow']['experiment_name']).experiment_id

    runs = mlflow_client.search_runs(
        experiment_ids=[exp_id],
        filter_string="attributes.run_name = '{}'".format(run_name),
    )
    assert len(runs) == 1
    run_id = runs[0].info.run_id

    for k, v in extra_params.items():
        mlflow_client.log_param(run_id, k, v)


def run(fine_tune_config):
    run_name = fine_tune_config['run_name']
    working_dir = fine_tune_config['working_dir']
    out_fname = fine_tune_config['out_fname']

    exp_name = fine_tune_config['mlflow']['experiment_name']
    if fine_tune_config['method'] == 'from_scratch':
        from_scratch(
            small_dt_name=fine_tune_config['small_dt_name'], run_name=run_name, exp_name=exp_name,
            out_fname=out_fname,
            train_batch_size=fine_tune_config['train_batch_size'], epochs=fine_tune_config['epochs'],
            learning_rate=fine_tune_config['learning_rate'], n_layers=fine_tune_config['n_layers'],
            hidden_size=fine_tune_config['hidden_size'],
        )
        return get_from_scratch_config_dict(
            fine_tune_config['epochs'], exp_name, fine_tune_config['hidden_size'],
            fine_tune_config['learning_rate'], fine_tune_config['n_layers'], run_name,
            fine_tune_config['small_dt_name'], fine_tune_config['train_batch_size'])
    elif fine_tune_config['method'] == 'progressive':
        if fine_tune_config.get('ewc_lambda') is not None:
            ewc_fname = 'ewc.pth'
            fit_ewc_fork(
                config_dict_fname=fine_tune_config['config_dict_fname'],
                input_checkpoint_fname=fine_tune_config['input_checkpoint_fname'],
                large_dt_name=fine_tune_config['large_dt_name'],
                ewc_lambda=fine_tune_config['ewc_lambda'],
                batch_size=fine_tune_config['train_batch_size'],
                output_fname=ewc_fname
            )
        else:
            ewc_fname = None

        progressive_unfreeze(
            working_dir=working_dir, input_checkpoint_fname=fine_tune_config['input_checkpoint_fname'],
            config_dict_fname=fine_tune_config['config_dict_fname'],
            large_dt_name=fine_tune_config['large_dt_name'], small_dt_name=fine_tune_config['small_dt_name'],
            run_name=run_name, exp_name=exp_name,
            train_batch_size=fine_tune_config['train_batch_size'], total_layers=fine_tune_config['n_layers'],
            epochs_per_layer=fine_tune_config['epochs_per_layer'], final_epochs=fine_tune_config['final_epochs'],
            learning_rate=fine_tune_config['learning_rate'],
            ewc_fname=ewc_fname,
            out_fname=out_fname,
        )

    elif fine_tune_config['method'] == 'only_embeddings':
        fine_tune_embeddings(
            working_dir=working_dir, input_checkpoint_fname=fine_tune_config['input_checkpoint_fname'],
            config_dict_fname=fine_tune_config['config_dict_fname'],
            large_dt_name=fine_tune_config['large_dt_name'], small_dt_name=fine_tune_config['small_dt_name'],
            run_name=run_name, exp_name=exp_name,
            train_batch_size=fine_tune_config['train_batch_size'],
            final_epochs=fine_tune_config['final_epochs'],
            learning_rate=fine_tune_config['learning_rate'],
            out_fname=out_fname,
        )

    elif fine_tune_config['method'] in ('full', 'ewc'):
        if fine_tune_config['method'] == 'ewc':
            ewc_fname = 'ewc.pth'
            fit_ewc_fork(
                config_dict_fname=fine_tune_config['config_dict_fname'],
                input_checkpoint_fname=fine_tune_config['input_checkpoint_fname'],
                large_dt_name=fine_tune_config['large_dt_name'],
                ewc_lambda=fine_tune_config['ewc_lambda'],
                batch_size=fine_tune_config['train_batch_size'],
                output_fname=ewc_fname
            )
        else:
            ewc_fname = None

        full_fine_tune(
            working_dir=working_dir, config_dict_fname=fine_tune_config['config_dict_fname'],
            input_checkpoint_fname=fine_tune_config['input_checkpoint_fname'],
            out_fname=out_fname,
            large_dt_name=fine_tune_config['large_dt_name'], small_dt_name=fine_tune_config['small_dt_name'],
            run_name=run_name, exp_name=exp_name,
            learning_rate=fine_tune_config['learning_rate'], epochs=fine_tune_config['epochs'],
            train_batch_size=fine_tune_config['train_batch_size'],
            ewc_fname=ewc_fname,
            freeze_embeddings=fine_tune_config['freeze_embeddings'],
            unfreeze_n_layers=fine_tune_config['unfreeze_n_layers'],
        )


def fine_tune_embeddings(*, working_dir, input_checkpoint_fname, small_dt_name, config_dict_fname,
                         large_dt_name, run_name, exp_name, final_epochs,
                         learning_rate, train_batch_size, out_fname):
    fs.ensure_exists(working_dir)

    small_dt_fname = fs.join(working_dir, small_dt_name + '.pkl')
    chkp_dir = fs.ensure_exists(fs.join(working_dir, 'checkpoints'))
    adapted_checkpoint_fname = fs.join(chkp_dir, 'adapted_model.pth')

    adapt(config_dict_fname=config_dict_fname,
          input_checkpoint_fname=input_checkpoint_fname,
          run_name=run_name,
          exp_name=exp_name,
          large_dt_name=large_dt_name,
          output_checkpoint_fname=adapted_checkpoint_fname,
          output_dataset_fname=small_dt_fname,
          small_dt_name=small_dt_name)

    prev_checkpoint_fname = adapted_checkpoint_fname
    fine_tune(
        config_dict_fname=config_dict_fname,
        small_dt_fname=small_dt_fname,
        input_checkpoint_fname=prev_checkpoint_fname,
        epochs=final_epochs,
        freeze_embeddings=False,
        unfreeze_n_layers=0,
        output_checkpoint_fname=out_fname,
        run_name=run_name, exp_name=exp_name,
        train_batch_size=train_batch_size, learning_rate=learning_rate,
    )


def progressive_unfreeze(*, working_dir, input_checkpoint_fname, small_dt_name, config_dict_fname,
                         large_dt_name, run_name, exp_name, epochs_per_layer, final_epochs, total_layers,
                         learning_rate, train_batch_size, ewc_fname, out_fname):
    fs.ensure_exists(working_dir)

    small_dt_fname = fs.join(working_dir, small_dt_name + '.pkl')
    chkp_dir = fs.ensure_exists(fs.join(working_dir, 'checkpoints'))
    adapted_checkpoint_fname = fs.join(chkp_dir, 'adapted_model.pth')

    adapt(config_dict_fname=config_dict_fname,
          input_checkpoint_fname=input_checkpoint_fname,
          run_name=run_name,
          exp_name=exp_name,
          large_dt_name=large_dt_name,
          output_checkpoint_fname=adapted_checkpoint_fname,
          output_dataset_fname=small_dt_fname,
          small_dt_name=small_dt_name)

    prev_checkpoint_fname = adapted_checkpoint_fname
    for i in range(1, total_layers + 1):
        output_checkpoint_fname = fs.join(chkp_dir, f'{i:02d}.pth')
        fine_tune(
            config_dict_fname=config_dict_fname,
            small_dt_fname=small_dt_fname,
            input_checkpoint_fname=prev_checkpoint_fname,
            epochs=epochs_per_layer,
            freeze_embeddings=True,
            unfreeze_n_layers=i,
            output_checkpoint_fname=output_checkpoint_fname,
            run_name=run_name, exp_name=exp_name,
            train_batch_size=train_batch_size, learning_rate=learning_rate,
            ewc_fname=ewc_fname
        )
        prev_checkpoint_fname = output_checkpoint_fname

    fine_tune(
        config_dict_fname=config_dict_fname,
        small_dt_fname=small_dt_fname,
        input_checkpoint_fname=prev_checkpoint_fname,
        epochs=final_epochs,
        freeze_embeddings=False,
        unfreeze_n_layers=-1,
        output_checkpoint_fname=out_fname,
        run_name=run_name, exp_name=exp_name,
        train_batch_size=train_batch_size, learning_rate=learning_rate,
        ewc_fname=ewc_fname
    )


def full_fine_tune(*, working_dir, config_dict_fname, input_checkpoint_fname, run_name, exp_name, large_dt_name,
                   out_fname, small_dt_name,
                   # fine-tune-specific-args
                   epochs, train_batch_size, learning_rate, ewc_fname=None,
                   freeze_embeddings=False, unfreeze_n_layers=-1):
    fs.ensure_exists(working_dir)
    chkp_dir = fs.ensure_exists(fs.join(working_dir, 'checkpoints'))
    small_dt_fname = fs.join(working_dir, small_dt_name + '.pkl')
    adapted_checkpoint_fname = fs.join(chkp_dir, 'adapted_model.pth')

    adapt(
        config_dict_fname=config_dict_fname,
        input_checkpoint_fname=input_checkpoint_fname,
        run_name=run_name,
        exp_name=exp_name,
        large_dt_name=large_dt_name,
        output_checkpoint_fname=adapted_checkpoint_fname,
        output_dataset_fname=small_dt_fname,
        small_dt_name=small_dt_name
    )

    fine_tune(
        config_dict_fname=config_dict_fname,
        small_dt_fname=small_dt_fname,
        input_checkpoint_fname=adapted_checkpoint_fname,
        epochs=epochs,
        freeze_embeddings=freeze_embeddings,
        unfreeze_n_layers=unfreeze_n_layers,
        output_checkpoint_fname=out_fname,
        run_name=run_name, exp_name=exp_name,
        train_batch_size=train_batch_size, learning_rate=learning_rate,
        ewc_fname=ewc_fname
    )


def from_scratch(*, small_dt_name, epochs, out_fname, train_batch_size, learning_rate, hidden_size,
                 n_layers, run_name, exp_name):
    p = Process(
        target=train_model_from_scratch,
        kwargs=dict(
            small_dt_name=small_dt_name, epochs=epochs, out_fname=out_fname, train_batch_size=train_batch_size,
            learning_rate=learning_rate, hidden_size=hidden_size,
            n_layers=n_layers, run_name=run_name, exp_name=exp_name
        )
    )
    p.start()
    p.join()


def fit_ewc_fork(*, config_dict_fname, input_checkpoint_fname, large_dt_name, ewc_lambda, batch_size, output_fname):
    p = Process(
        target=fit_ewc,
        kwargs=dict(
            config_dict_fname=config_dict_fname,
            input_checkpoint_fname=input_checkpoint_fname,
            large_dt_name=large_dt_name,
            ewc_lambda=ewc_lambda,
            output_fname=output_fname,
            batch_size=batch_size
        )
    )
    p.start()
    p.join()


def adapt(*, config_dict_fname, input_checkpoint_fname, run_name, exp_name, large_dt_name,
          output_checkpoint_fname, output_dataset_fname, small_dt_name):
    p = Process(
        target=adapt_model,
        kwargs=dict(
            config_dict_fname=config_dict_fname,
            input_checkpoint_fname=input_checkpoint_fname,
            run_name=run_name,
            exp_name=exp_name,
            large_dt_name=large_dt_name,
            output_checkpoint_fname=output_checkpoint_fname,
            output_dataset_fname=output_dataset_fname,
            small_dt_name=small_dt_name
        )
    )
    p.start()
    p.join()


def fine_tune(*, config_dict_fname, small_dt_fname, input_checkpoint_fname,
              epochs, freeze_embeddings, unfreeze_n_layers, output_checkpoint_fname,
              run_name, exp_name, train_batch_size=None, learning_rate=None, ewc_fname=None):
    p = Process(
        target=fine_tune_model,
        kwargs=dict(
            config_dict_fname=config_dict_fname,
            small_dt_fname=small_dt_fname,
            input_checkpoint_fname=input_checkpoint_fname,
            epochs=epochs,
            freeze_embeddings=freeze_embeddings,
            unfreeze_n_layers=unfreeze_n_layers,
            output_checkpoint_fname=output_checkpoint_fname,
            run_name=run_name,
            exp_name=exp_name,
            train_batch_size=train_batch_size,
            learning_rate=learning_rate,
            ewc_fname=ewc_fname
        )
    )
    p.start()
    p.join()


if __name__ == '__main__':
    main()
