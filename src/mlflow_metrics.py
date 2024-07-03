import pandas as pd
from IPython.core.display import display
from mlflow import MlflowClient
from tqdm.notebook import tqdm


class MetricsManager:
    def __init__(self, tracking_uri):
        self.mlflow_client = MlflowClient(tracking_uri)

    def list_experiments(self):
        return sorted([(int(e.experiment_id), e.name) for e in self.mlflow_client.search_experiments()])

    def list_runs(self, exp_id, *extra_fields, verbose=False):
        exp_id = str(exp_id)
        runs = self.mlflow_client.search_runs(experiment_ids=[exp_id])
        docs = []
        res = []
        for i, r in enumerate(runs):
            #         if r.data.metrics.get('epoch_tr_mtl') is None: continue
            res.append(r)
            doc = dict(
                i=i, run_id=r.info.run_id,
                run_name=r.info.run_name,
            )
            for m in extra_fields:
                attr = m.split('.')[0]
                key = m[len(attr) + 1:]
                doc[m] = getattr(r.data, attr).get(key)
            docs.append(doc)
        if verbose:
            display(pd.DataFrame(docs))
        return res

    def load_metrics(self, run, only_hr=False):
        metrics = {}
        for metric in run.data.metrics:
            if only_hr and '_hr_' not in metric:
                metrics[metric] = None
            else:
                s = self.get_metric(run, metric)
                s.name = metric
                metrics[metric] = s
        return metrics

    def get_metric(self, run, name, mode='step'):
        assert mode in ('step', 'ts', 'rel_ts', 'index')
        raw = self.mlflow_client.get_metric_history(run.info.run_id, name)
        if mode == 'step':
            return pd.Series([e.value for e in raw], index=[1 + e.step for e in raw]).sort_index()
        elif mode == 'ts':
            return pd.Series([e.value for e in raw], index=[e.timestamp for e in raw]).sort_index()
        elif mode == 'rel_ts':
            return pd.Series([e.value for e in raw], index=[e.timestamp - raw[0].timestamp for e in raw]).sort_index()
        else:
            return pd.Series([e.value for e in raw], index=list(range(len(raw)))).sort_index()

    def load_flops_metrics(self, runs, metric_name='valid/ndcg_at_5', early_stopping=False, use_knn_flops=True):
        skipped = 0
        series = []
        for run in tqdm(runs):
            metric = self.get_metric(run, metric_name).drop_duplicates()
            if 'total_flops' not in run.data.params:
                skipped += 1
                continue
            epoch_flops = (float(run.data.params['total_flops']) * int(run.data.params['n_inters']) // int(run.data.params['train_batch_size']))

            if run.data.params['nce_sampling_strategy'] == 'knn' and use_knn_flops:
                knn_flops = (
                        epoch_flops * self.get_metric(run, 'time_on_epoch_start').drop_duplicates()[1:].mean()
                        / self.get_metric(run, 'time_epoch').drop_duplicates()[1:].mean()
                )
            else:
                knn_flops = 0

            metric.index = metric.index * (epoch_flops + knn_flops)
            series.append(metric)
        if skipped:
            print('skipped (no flops data):', skipped)

        if early_stopping:
            for i, s in enumerate(series):
                #     new_s = s.rolling(3).max()
                #     new_s.iloc[:2] = s.iloc[:2]
                #     new_series.append(new_s)
                #     series[i] = new_s
                #     series[-50].plot()
                mask = (s.diff() / s) < -0.01
                if mask.sum():
                    series[i] = s[:s.index[mask][0]]

        return series
