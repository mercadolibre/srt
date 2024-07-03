import re
from copy import deepcopy

from frozendict import frozendict

from recbole.utils import InputType


class ConfigManager:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.config_dict = {
            'model': model, 'dataset': dataset,
            'load_col': {},
            'USER_ID_FIELD': self.user_id_field,
            'ITEM_ID_FIELD': self.item_id_field,

            'topk': 5,  # para metricas para reportar HR@5
            'valid_metric': 'ndcg@5',
            'mlflow': {
                'enable': False,
            },
        }
        self.add_inter_features()
        self.add_item_features()
        self.specify_architecture()
        self.specify_optimization()
        self.add_evaluation_args()

    def copy(self):
        res = ConfigManager(self.dataset, self.model)
        res.config_dict = deepcopy(self.config_dict)
        return res

    def mlfow(self, experiment_name, run_name, tracking_uri='mlruns'):
        self.config_dict['mlflow']['enable'] = True
        self.config_dict['mlflow']['experiment_name'] = experiment_name
        self.config_dict['mlflow']['run_name'] = run_name
        self.config_dict['mlflow']['tracking_uri'] = tracking_uri
        return self

    def override(self, **overrides):
        self.config_dict.update(overrides)
        return self

    def specify_scheduler(self, scheduler_type='one-cycle', **kwargs):
        self.config_dict['scheduler'] = {'type': scheduler_type, **kwargs}
        return self

    def specify_optimization(self, learning_rate=1e-3, weight_decay=1e-5,
                             max_grad_norm=1.0, loss_type='CE',
                             train_batch_size=128, eval_batch_size=128,
                             epochs=100, stopping_step=100):
        self.config_dict['learning_rate'] = learning_rate
        self.config_dict['weight_decay'] = weight_decay
        self.config_dict['max_grad_norm'] = max_grad_norm
        self.config_dict['loss_type'] = loss_type
        self.config_dict['train_batch_size'] = train_batch_size
        self.config_dict['eval_batch_size'] = eval_batch_size
        self.config_dict['epochs'] = epochs
        self.config_dict['stopping_step'] = stopping_step
        return self

    def specify_architecture(self, hidden_dropout_prob=0.2, attn_dropout_prob=0.2,
                             max_item_list_length=50, n_layers=2, hidden_size=50,
                             inner_size=50):

        self.config_dict['hidden_dropout_prob'] = hidden_dropout_prob
        self.config_dict['attn_dropout_prob'] = attn_dropout_prob
        self.config_dict['MAX_ITEM_LIST_LENGTH'] = max_item_list_length
        self.config_dict['n_layers'] = n_layers
        self.config_dict['hidden_size'] = hidden_size
        self.config_dict['inner_size'] = inner_size
        return self

    def add_evaluation_args(self, split=frozendict(LS='valid_and_test'), order='TO',
                            group_by='user', mode=frozendict(valid='full', test='full', train='full')):
        # https://recbole.io/docs/user_guide/config/evaluation_settings.html
        self.config_dict['eval_args'] = {
            'split': split, 'order': order, 'group_by': group_by, 'mode': mode
        }
        return self

    def add_negative_sampling_args(self, model_input_type=InputType.POINTWISE,
                                   distribution='popularity', sample_num=5,
                                   alpha=0.75, dynamic=False, candidate_num=10,
                                   co_counts_candidates=50, min_co_count=1, **kwargs):
        self.config_dict['MODEL_INPUT_TYPE'] = model_input_type
        self.config_dict['train_neg_sample_args'] = {
            'distribution': distribution, 'sample_num': sample_num,
            'alpha': alpha, 'dynamic': dynamic, 'candidate_num': candidate_num,
            'co_counts_candidates': co_counts_candidates, 'min_co_count': min_co_count,
            **kwargs
        }
        return self

    def no_negative_sampling(self):
        self.config_dict['train_neg_sample_args'] = None
        return self

    @property
    def is_feature_based(self):
        return self.model in ['SASRecF2', 'SASRecFNS']

    def add_inter_features(self):
        if 'meli_purchs' in self.dataset:
            features = ['user_num', 'item_num', 'timestamp', 'price']
        else:
            d = {
                'ml-1m': ['user_id', 'item_id', 'rating', 'timestamp'],
                'ml-100k': ['user_id', 'item_id', 'rating', 'timestamp'],
                '^amazon-beauty.*$': ['user_id', 'item_id', 'rating', 'timestamp'],
                '^amazon.*$': ['user_id', 'item_id', 'rating', 'timestamp'],
                '^pruned-amazon.*$': ['user_id', 'item_id', 'rating', 'timestamp'],
                'mcauley': ['user_id', 'item_id', 'rating', 'timestamp'],
                'amazon-toys-games': ['user_id', 'item_id', 'rating', 'timestamp'],
            }
            for pat, features in d.items():
                if re.match(pat, self.dataset):
                    break
            else:
                raise ValueError(f'Unknown dataset {self.dataset}')

        self.config_dict['load_col']['inter'] = features
        return self

    @property
    def item_id_field(self):
        if 'meli_purchs' in self.dataset:
            return 'item_num'
        else:
            return 'item_id'

    @property
    def user_id_field(self):
        if 'meli_purchs' in self.dataset:
            return 'user_num'
        else:
            return 'user_id'

    def add_item_features(self):
        if 'meli_purchs' in self.dataset:
            features = ['ITE_ITEM_TITLE', 'DOM_DOMAIN_ID']
        else:
            item_features_by_dataset = {
                'ml-1m': ['movie_title', 'release_year', 'genre'],
                'ml-100k': ['movie_title', 'release_year', 'class'],
                '^amazon-beauty.*$': ['title', 'brand'],  # 'sales_type', 'sales_rank', 'categories', 'price', '
                '^amazon.*$': ['title', 'brand'],  # 'sales_type', 'sales_rank', 'categories', 'price', '
                '^pruned-amazon.*$': ['title', 'brand'],  # 'sales_type', 'sales_rank', 'categories', 'price', '
                'amazon-toys-games': ['title', 'brand'],  # 'sales_type', 'sales_rank', 'categories', 'price', '
            }
            for pat, features in item_features_by_dataset.items():
                if re.match(pat, self.dataset):
                    break
            else:
                if self.is_feature_based:
                    raise ValueError(f'Feature based model {self.model} not supported for dataset {self.dataset}')
                else:
                    return self

        self.config_dict['load_col']['item'] = [self.item_id_field] + features

        if self.is_feature_based:
            self.config_dict['selected_features'] = features
        return self
