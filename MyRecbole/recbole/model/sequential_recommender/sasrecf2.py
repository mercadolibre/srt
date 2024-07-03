# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:32
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
SASRecF
################################################
"""
import gc

import faiss
import torch
from torch import nn
from torch.nn import NLLLoss
import numpy as np

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
from recbole.sampler import RepeatableSampler
from recbole.sampler.sampler import AbstractSampler
from recbole.utils import FeatureType
from src.progress import progress
from src.time_it import timeit
from pynvml.smi import nvidia_smi


class SASRecF2(SequentialRecommender):
    """This is an extension of SASRec, which concatenates item representations and item attribute representations
    as the input to the model.
    """

    def __init__(self, config, dataset):
        super(SASRecF2, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.embedding_size = config["embedding_size"] or config['hidden_size']  # for retrocompatibility
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.selected_features = config["selected_features"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]

        for field in self.selected_features:
            if field not in dataset.field2type:
                raise RuntimeError(
                    f'Invalid field {field} in selected_features! Available: {dataset.field2type.keys()}'
                )

        self.num_feature_field = sum(
            1
            if dataset.field2type[field] != FeatureType.FLOAT_SEQ
            else dataset.num(field)
            for field in config["selected_features"]
        )

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset,
            self.embedding_size,
            self.selected_features,
            self.pooling_mode,
            self.device,
        )
        self.normalize_embedding = config["normalize_embedding"]

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.adapter_layer = nn.Linear(
            self.embedding_size * self.num_feature_field, self.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.item_num = dataset.item_num

        if self.loss_type == "COS":
            self.loss_fct = nn.CosineEmbeddingLoss(reduction='mean')
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == "NS":
            self.loss_fct = nn.BCEWithLogitsLoss()
        elif self.loss_type == 'NS2':
            self.sampler = RepeatableSampler('train', dataset)
            self.loss_fct = nn.BCEWithLogitsLoss()
        elif self.loss_type == 'InfoNCE':
            self.sampler = RepeatableSampler('train', dataset)
            self.num_negatives = config['nce_num_negatives']
            # self.sampler = CoCountsSampler(dataset, n_candidates=50, min_co_count=5)
            # self.sampler._build_co_counts_table()
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == 'InfoNCE-quick':
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=config['nce_label_smoothing'])
            self.num_negatives = config['nce_num_negatives']
            self.temperature = config['nce_temperature']
            self.global_negatives = config['nce_global_negatives']
            self.sampling_strategy = config['nce_sampling_strategy']
            assert self.sampling_strategy in ['uniform', 'popularity', 'log_popularity', 'co-counts', 'similarity',
                                              'positive-mixing', 'knn']
            if self.sampling_strategy == 'similarity': assert not self.global_negatives

            if self.sampling_strategy != 'uniform':
                self.update_item_distr(dataset)

            if self.sampling_strategy == 'positive-mixing':
                self.mixing_factor = config['positive_mixing_factor']

            elif self.sampling_strategy == 'knn':
                self.nce_knn_hard_negatives_pool_size = config['nce_knn_hard_negatives_pool_size']
                self.nce_knn_discard_neighbors = config['nce_knn_discard_neighbors']
                self.nce_knn_tot_neighbors = self.nce_knn_hard_negatives_pool_size + self.nce_knn_discard_neighbors
                self.nce_knn_random_frac = config['nce_knn_random_frac']

            elif self.sampling_strategy == 'co-counts':
                self.mixing_factor = config['co_counts_mixing_factor']
                with timeit('build co-counts table'):
                    self._build_co_counts_table(dataset)
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['COS', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ["feature_embed_layer"]

        self._item_features = None
        self._item_features_version = None

        self.epoch = 0

    def update_item_distr(self, dataset):
        pop_table = dataset.item_popularity_distr
        assert len(pop_table) == max(pop_table.keys())
        _, freqs = zip(*sorted(pop_table.items()))
        # Add 0 probability for the padding item
        freqs = torch.cat([torch.tensor([0.0]), torch.tensor(freqs)]).to(self.device)
        if self.sampling_strategy == 'log_popularity':
            freqs = torch.log(freqs + 1)
        self.item_distr = freqs / freqs.sum()
        self.item_num = dataset.item_num

    def get_learning_rates(self, base_learning_rate):
        res = [
            {'params': self.feature_embed_layer.parameters(), 'lr': base_learning_rate * 0.1},
            {'params': self.position_embedding.parameters(), 'lr': base_learning_rate * 0.1},
            {'params': self.adapter_layer.parameters(), 'lr': base_learning_rate * 0.1},
            {'params': self.LayerNorm.parameters(), 'lr': base_learning_rate * 0.1},
        ]

        for i, layer in enumerate(self.trm_encoder.layers):
            res.append({'params': layer.parameters(), 'lr': base_learning_rate * (i + 1) / self.n_layers})

        return res

    def get_item_features_table(self):
        return self.embed_items(torch.arange(self.item_num).to(self.device))

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        # position embedding
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = self.embed_items(item_seq)

        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output  # [B H]

    def embed_items(self, item_seq):
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding["item"]
        dense_embedding = dense_embedding["item"]
        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        feature_table = torch.cat(feature_table, dim=-2)
        table_shape = feature_table.shape
        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(
            table_shape[:-2] + (feat_num * embedding_size,)
        )
        res = self.adapter_layer(feature_emb)
        if self.normalize_embedding:
            res = res / res.norm(2, dim=1)[:, None]
        return res

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)  # [B H]
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "COS":
            pos_items_emb = self.embed_items(pos_items)
            loss = self.loss_fct(seq_output, pos_items_emb, interaction["label"])
            return loss
        elif self.loss_type == 'NS':
            items_emb = self.embed_items(pos_items)
            logits = torch.mul(items_emb, seq_output).sum(1)
            return self.loss_fct(logits, interaction['label'])
        elif self.loss_type == 'CE':
            test_item_emb = self.get_item_features_table()
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss
        elif self.loss_type == 'InfoNCE':
            pos_items_emb = self.embed_items(pos_items)
            neg_items_emb = self.embed_items(self.sampler.sample_by_user_ids(pos_items, pos_items, self.num_negatives))
            # neg_items_emb = self.embed_items(self.sampler.sample_by_co_counts(interaction, self.num_negatives))
            pos_logits = (seq_output * pos_items_emb).sum(1)
            neg_logits = (seq_output.repeat(self.num_negatives, 1) * neg_items_emb).sum(1)
            bs = pos_items.size(0)
            logits = torch.cat([pos_logits.view(bs, -1), neg_logits.view(bs, -1)], dim=1)
            loss = self.loss_fct(logits, torch.zeros(logits.size(0), dtype=torch.long).to(self.device))
            return loss
        elif self.loss_type == 'InfoNCE-quick':
            bs = pos_items.size(0)
            pos_items_emb = self.embed_items(pos_items)  # [B, H]

            if self.sampling_strategy == 'uniform':
                if self.global_negatives:
                    neg_item_ids = torch.randint(1, self.item_num, (self.num_negatives,)).to(self.device)
                else:
                    neg_item_ids = torch.randint(1, self.item_num, (bs, self.num_negatives)).to(self.device)

                neg_probs = torch.ones_like(neg_item_ids, dtype=torch.float32) / self.item_num
                pos_probs = torch.ones_like(pos_items, dtype=torch.float32) / self.item_num
            elif self.sampling_strategy == 'similarity':
                sims = (seq_output / seq_output.norm(dim=1)[:, None]) @ (
                        pos_items_emb / pos_items_emb.norm(dim=1)[:, None]).t()  # [B, B]
                sims = (sims + 1) / 2  # make it positive
                sims.fill_diagonal_(0)
                sims = sims / sims.sum(1)[:, None]
                indices = torch.multinomial(sims, self.num_negatives)
                ##                indices = sims.argsort(dim=1, descending=True)[:, :self.num_negatives]
                rows = torch.arange(bs).repeat_interleave(self.num_negatives).to(self.device)
                neg_item_ids = pos_items[indices]  # [B, num_negatives]
                neg_probs = sims[rows, indices.reshape(-1)].reshape(bs, -1)  # [B, num_negatives]
                pos_probs = torch.ones_like(pos_items, dtype=torch.float32) / self.item_num
                neg_probs = torch.ones_like(neg_item_ids, dtype=torch.float32) / self.item_num
            elif self.sampling_strategy in ('co-counts', 'popularity', 'log-popularity'):
                item_distr = self._get_item_distr(interaction)

                if self.global_negatives:
                    neg_item_ids = (
                        torch.multinomial(item_distr, self.num_negatives, replacement=True).to(self.device)
                    )
                else:
                    neg_item_ids = (
                        torch.multinomial(item_distr, self.num_negatives * bs, replacement=True)
                        .view(bs, -1).to(self.device)
                    )
                neg_probs = self.item_distr[neg_item_ids]
                pos_probs = self.item_distr[pos_items]

            elif self.sampling_strategy == 'knn':
                assert not self.global_negatives

                pos_probs = self.item_distr[pos_items]
                n_hard = int(self.num_negatives * (1 - self.nce_knn_random_frac))
                n_random = self.num_negatives - n_hard

                if n_hard:
                    neg_item_ids = torch.gather(
                        self.hard_negatives[pos_items], 1,
                        torch.randint(0, self.nce_knn_hard_negatives_pool_size, size=(len(pos_items), n_hard)).to(
                            'cuda')
                    )

                    #                     neg_probs = torch.ones_like(neg_item_ids,
                    #                                                 dtype=torch.float32) * self.num_negatives / self.nce_knn_hard_negatives_pool_size
                    #                     neg_probs = torch.ones_like(neg_item_ids,
                    #                                                 dtype=torch.float32) / self.nce_knn_hard_negatives_pool_size
                    neg_probs = self.item_distr[neg_item_ids]

                if n_random:
                    new_neg_item_ids = (
                        torch.multinomial(
                            self.item_distr, n_random * bs, replacement=True
                        ).view(bs, -1).to(self.device)
                    )

                    if n_hard:
                        neg_item_ids = torch.cat([new_neg_item_ids, neg_item_ids], dim=1)

                        neg_probs = torch.cat([neg_probs, self.item_distr[new_neg_item_ids]], dim=1)
                    else:
                        neg_item_ids = new_neg_item_ids
                        neg_probs = self.item_distr[new_neg_item_ids]

            else:
                raise RuntimeError('invalid sampling strategy')

            # neg_item_ids has shape [num_negatives] if global_negatives else [B, num_negatives]
            if self.sampling_strategy == 'similarity':
                neg_items_emb = pos_items_emb[indices]
            else:
                neg_items_emb = self.embed_items(neg_item_ids)  # [num_negatives, H] or [B, num_negatives, H]

            global_negatives = self.global_negatives
            if self.sampling_strategy == 'positive-mixing':
                assert self.global_negatives
                global_negatives = False  # loss computation must be done with local negatives

                #                 directions = (neg_items_emb[neg_1] - neg_items_emb[neg_2]).detach()
                if not hasattr(self, 'directions'):
                    directions = torch.randn((bs, self.num_negatives, pos_items_emb.shape[1])).to('cuda')
                    directions = directions * ((torch.randn((bs, self.num_negatives)).to(
                        'cuda').abs() * 0.375099 + 0.671044).abs() / directions.norm(dim=2))[:, :, None]
                    self.directions = directions
                elif self.directions.shape[0] != bs:
                    directions = torch.randn((bs, self.num_negatives, pos_items_emb.shape[1])).to('cuda')
                    directions = directions * ((torch.randn((bs, self.num_negatives)).to(
                        'cuda').abs() * 0.3875099 + 0.6471044).abs() / directions.norm(dim=2))[:, :, None]
                else:
                    directions = self.directions

                #                 norm = pos_items_emb.norm(dim=1)[:, None, None].repeat(1, self.num_negatives, 1) / directions.norm(dim=2)[:, :, None]
                directions = directions * self.mixing_factor
                if torch.isnan(directions).sum() > 0: import ipdb;ipdb.set_trace()
                neg_items_emb = pos_items_emb[:, None, :].repeat(1, self.num_negatives, 1) + directions

                # a = self.mixing_factor
                # neg_items_emb = neg_items_emb.repeat(bs, 1, 1)
                # neg_items_emb = a * pos_items_emb[:, None, :].repeat(1, self.num_negatives, 1) + (1 - a) * neg_items_emb

            if global_negatives:
                # neg_items_emb is [num_negatives, H]
                # seq_output is [B, H]
                neg_logits = seq_output @ neg_items_emb.t()  # [B, num_negatives]
            else:
                neg_logits = (
                        seq_output[:, None, :].repeat(1, self.num_negatives, 1) * neg_items_emb
                ).sum(2)  # [B, num_negatives]

            pos_logits = (seq_output * pos_items_emb).sum(1)  # [B]

            # logQ correction
            epsilon = 1e-16
            neg_logits -= torch.log(neg_probs + epsilon)
            pos_logits -= torch.log(pos_probs + epsilon)

            # adjust the cases when the target was sampled
            # value taken from https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/2c9ab4093a8367e02943f39174de2cd5720cb91e/transformers4rec/torch/model/prediction_task.py#L535
            value = torch.finfo(torch.float16).min / 100
            neg_logits[pos_items.reshape(-1, 1).repeat(1, self.num_negatives) == neg_item_ids] = value

            logits = torch.cat([pos_logits.view(bs, -1), neg_logits.view(bs, -1)], dim=1) / self.temperature
            loss = self.loss_fct(logits, torch.zeros(logits.size(0), dtype=torch.long).to(self.device))
            return loss

        elif self.loss_type == 'NS2':
            pos_items_emb = self.embed_items(pos_items)
            neg_items_emb = self.embed_items(self.sampler.sample_by_user_ids(pos_items, pos_items, self.num_negatives))
            pos_logits = (seq_output * pos_items_emb).sum(1)
            neg_logits = (seq_output.repeat(self.num_negatives, 1) * neg_items_emb).sum(1)
            label = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0).to(self.device)
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            loss = self.loss_fct(logits, label)
            return loss

    def _get_item_distr(self, interaction):
        if self.sampling_strategy == 'co-counts':
            history = interaction.item_num_list  # TODO: item_num for meli item_id for the rest...
            related = self.co_counts_table[history].view(-1).unique()[1:]

            item_distr = self.item_distr[:]
            item_distr[related] = item_distr[related] * self.mixing_factor / item_distr[related].sum()
            not_related = torch.ones(item_distr.shape, dtype=torch.bool, device=self.device)
            not_related[related] = False
            item_distr[not_related] = item_distr[not_related] * (1 - self.mixing_factor) / item_distr[not_related].sum()
        else:
            item_distr = self.item_distr
        return item_distr

    def _build_co_counts_table(self, train_set):
        co_counts = train_set.get_co_counts()
        # TODO: parametrize
        self.n_candidates = 30
        self.min_co_count = 2

        self.co_counts_table = torch.zeros((self.item_num, self.n_candidates), dtype=torch.int32)
        for iid, co_counts in co_counts.items():
            top_co_counts = sorted(co_counts.items(), key=lambda x: (-x[1], x[0]))
            for i, (co_iid, co_count) in enumerate(top_co_counts):
                if co_count < self.min_co_count: break
                if i >= self.n_candidates: break
                self.co_counts_table[iid, i] = co_iid

        self.co_counts_table = self.co_counts_table.to(self.device)

    def on_epoch_start(self, return_similarities=False, increase_epoch=True):
        """
        return_similarities is used only for debugging purposes, and does not move hard negatives to cuda
        so it will not work when training the model
        """
        if self.loss_type != 'InfoNCE-quick' or self.sampling_strategy != 'knn': return
        if increase_epoch:
            self.epoch += 1

        # hard negatives are computed for flops estimation. When training is starting,
        # it is not necessary to compute them again
        if self.epoch == 1 and hasattr(self, 'hard_negatives'):
            assert not return_similarities
            return

        gc_cuda()
        print(f'[knn] free memory: {get_free_mem() / 1024:.02f}GB')

        with timeit('[knn] compute item embeddings'):
            items = []
            step = 2048  # it can be more
            for start in torch.arange(0, self.item_num, step):
                end = min(start + step, self.item_num)
                with torch.no_grad():
                    items.append(self.embed_items(torch.arange(start, end).to(self.device)).cpu())

            items = torch.cat(items)
            items = (items / items.norm(2, dim=1)[:, None]).numpy()

        faiss.omp_set_num_threads(8)
        with timeit('[knn] create index'):
            index = faiss.IndexFlatIP(items.shape[1])
            index.add(items)
            res = faiss.StandardGpuResources()  # use a single GPU
            # make it into a gpu index
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

        res.noTempMemory()
        gc_cuda()

        bs = 1024 * 4
        hard_negatives = torch.zeros((len(items), self.nce_knn_hard_negatives_pool_size), dtype=torch.int32)
        if return_similarities:
            similarities = torch.zeros((len(items), self.nce_knn_hard_negatives_pool_size), dtype=torch.float32)

        for start in progress(range(0, len(items), bs), desc='[knn] compute hard negatives'):
            end = min(len(items), start + bs)
            D, I = gpu_index.search(items[start:end], k=self.nce_knn_tot_neighbors + 1)
            I = torch.tensor(I)

            # remove trigger
            mask = (I != torch.arange(start, end)[:, None].repeat((1, self.nce_knn_tot_neighbors + 1)))
            mask[mask.sum(1) == self.nce_knn_tot_neighbors + 1, 0] = False
            I = I[mask].view((end - start, self.nce_knn_tot_neighbors))

            if self.nce_knn_discard_neighbors > 0:
                I = I[:, self.nce_knn_discard_neighbors:]

            if return_similarities:
                similarities[start: end] = (
                    torch.tensor(D)[mask].view((end - start, self.nce_knn_tot_neighbors))[:,
                    self.nce_knn_discard_neighbors:]
                )

            hard_negatives[start: end] = I

        if not return_similarities:
            self.hard_negatives = hard_negatives.to('cuda')

        index.reset()
        gpu_index.reset()
        gc_cuda()
        print(f'[knn] free memory: {get_free_mem() / 1024:.02f}GB')
        if return_similarities:
            return similarities

    def get_num_non_embedding_params(self):
        embedding_params = sum([np.prod(p.size()) for p in self.feature_embed_layer.parameters()])
        return self.get_num_params() - embedding_params

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.embed_items([test_item])  # [1, H]
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def sampled_predict(self, interaction, n_negatives):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        neg_item_ids = torch.randint(1, self.item_num, (n_negatives,))
        pos_item_ids = interaction[self.ITEM_ID]
        neg_item_embs = self.embed_items(neg_item_ids.to(self.device))
        pos_item_embs = self.embed_items(pos_item_ids.to(self.device))
        neg_scores = torch.matmul(seq_output, neg_item_embs.transpose(0, 1))
        pos_scores = torch.mul(seq_output, pos_item_embs).sum(dim=1)
        return torch.cat([pos_scores.reshape((-1, 1)), neg_scores], dim=1)

    def full_sort_predict(self, interaction, item_batch_size=50000):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        all_scores = []
        for start in range(0, self.item_num, item_batch_size):
            end = min(start + item_batch_size, self.item_num)
            test_items_emb = self.embed_items(torch.arange(start, end).to(self.device))
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
            all_scores.append(scores.cpu())

        return torch.cat(all_scores, dim=1)


def get_free_mem():
    nvsmi = nvidia_smi.getInstance()
    return nvsmi.DeviceQuery('memory.free')['gpu'][0]['fb_memory_usage']['free']


def gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
