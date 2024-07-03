# -*- coding: utf-8 -*-

from recbole.config import Config


class MLFlowLogger(object):
    def __init__(self, config: Config):
        """
        Args:
            config (dict): A dictionary of parameters used by RecBole.
        """
        self.config = config
        try:
            self.mlflow_config = config.mlflow
        except AttributeError:
            self.mlflow_config = {}

        self.enabled = self.mlflow_config.get('enable', False)
        self.continue_run = self.mlflow_config.get('auto_continue', False)
        self.epoch_offset = 0
        self.mlflow_client = self.exp_id = self.new_run = self.run_id = None
        self.setup()

    def setup(self):
        if self.enabled:
            self.mlflow_client = get_mlflow_client(self.mlflow_config)

            exp_name = self.mlflow_config['experiment_name']
            self.exp_id = get_or_create_experiment(self.mlflow_client, exp_name)

            if self.mlflow_config.get('auto_continue'):
                runs = self.mlflow_client.search_runs(
                    experiment_ids=[self.exp_id],
                    filter_string="attributes.run_name = '{}'".format(self.mlflow_config['run_name']),
                )

                assert len(runs) <= 1
                if len(runs) == 0:
                    run = self._create_run()
                else:
                    self.new_run = False
                    run = runs[0]
                    print(f'Continuing run {run.info.run_id}')
                    if 'train/epoch' in run.data.metrics:
                        self.epoch_offset = run.data.metrics['train/epoch'] + 1
            else:
                run = self._create_run()

            self.run_id = run.info.run_id

            if self.new_run:
                for k, v in self.config.final_config_dict.items():
                    self.mlflow_client.log_param(self.run_id, k, v)

    def _create_run(self):
        self.new_run = True
        run = self.mlflow_client.create_run(
            experiment_id=self.exp_id, run_name=self.mlflow_config['run_name'],
        )
        return run

    def log_param(self, key, value):
        if self.enabled:
            try:
                self.mlflow_client.log_param(self.run_id, key, value)
            except Exception as e:
                if self.new_run:
                    # should not happen for a new run
                    raise e
                else:
                    # could happen when resuming a run
                    print(f'tried to log param {key} with value {value} but failed')
                    # traceback.print_exc()

    def log_params(self, dict):
        if self.enabled:
            for k, v in dict.items():
                self.log_param(k, v)

    def log_model_params(self, model):
        if self.enabled:
            self.log_param("model_total_params", model.get_num_params())
            try:
                # log non embedding params if possible
                self.log_param("model_non_emb_params", model.get_num_non_embedding_params())
            except NotImplementedError:
                pass

    def finish_training(self, status):
        if self.enabled:
            self.mlflow_client.set_terminated(self.run_id, status)

    def log_metrics(self, metrics, epoch, head="train", tail='', commit=True):
        epoch += self.epoch_offset
        if self.enabled:
            if head:
                metrics = self._add_head_to_metrics(metrics, head, tail)

            for k, v in metrics.items():
                self.mlflow_client.log_metric(self.run_id, k, v, step=epoch)

    def log_metric(self, name, value, epoch):
        epoch += self.epoch_offset
        if self.enabled:
            self.mlflow_client.log_metric(self.run_id, name, value, step=epoch)

    def log_eval_metrics(self, metrics, epoch, head="eval", tail=''):
        epoch += self.epoch_offset
        if self.enabled:
            metrics = self._add_head_to_metrics(metrics, head, tail)
            for k, v in metrics.items():
                self.mlflow_client.log_metric(self.run_id, k, v, step=epoch)

    def _add_head_to_metrics(self, metrics, head, tail):
        tail = f'-{tail}' if tail else ''
        head_metrics = dict()
        for k, v in metrics.items():
            k = k.replace('@', '_at_')
            if "_step" in k:
                head_metrics[k] = v
            else:
                head_metrics[f"{head}/{k}{tail}"] = v

        return head_metrics


def flatten_dict(d):
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = prefix_dict(k, flatten_dict(v), sep='.')
            res.update(v)
        else:
            res[k] = v
    return res


def prefix_dict(p, d, sep='_'):
    return {f'{p}{sep}{k}': v for k, v in d.items()}


def get_mlflow_client(mlflow_config: dict):
    try:
        from mlflow.tracking import MlflowClient

    except ImportError:
        raise ImportError(
            "To use the MLFlow Logger please install mlflow."
            "Run `pip install mlflow` to install it."
        )
    return MlflowClient(mlflow_config['tracking_uri'])


def get_or_create_experiment(mlflow_client, exp_name):
    # get or create experiment
    try:
        return mlflow_client.create_experiment(exp_name)
    except:
        return mlflow_client.get_experiment_by_name(exp_name).experiment_id
