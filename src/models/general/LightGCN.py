import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel


class LightGCNBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of LightGCN layers.')
        parser.add_argument('--mix_prob', type=float, default=0.5,
                            help='Probability of applying IMix')
        parser.add_argument('--mix_alpha', type=float, default=1.0,
                            help='Alpha parameter for beta distribution in IMix')
        return parser

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()

        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1)) + 1e-10

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        if selfloop_flag:
            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        else:
            norm_adj_mat = normalized_adj_single(adj_mat)

        return norm_adj_mat.tocsr()

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.mix_prob = args.mix_prob  # IMix参数：混合概率
        self.mix_alpha = args.mix_alpha  # IMix参数：beta分布参数
        self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)

    def init_weights(self, module):
        if isinstance(module, nn.ParameterDict):
            for param in module.values():
                nn.init.xavier_uniform_(param.data)

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)

        # 训练阶段应用IMix
        if self.training and self.mix_prob > 0:
            batch_size = u_embed.size(0)
            # 随机决定哪些样本需要混合
            mask = torch.rand(batch_size, device=u_embed.device) < self.mix_prob
            if mask.any():
                # 生成混合系数
                lam = torch.distributions.Beta(self.mix_alpha, self.mix_alpha).sample((mask.sum(),)).to(u_embed.device)
                lam = torch.max(lam, 1 - lam)  # 确保lam >= 0.5

                # 生成随机打乱的索引用于混合
                rand_idx = torch.randperm(batch_size, device=u_embed.device)

                # 混合用户嵌入
                u_mix = lam.view(-1, 1) * u_embed[mask] + (1 - lam.view(-1, 1)) * u_embed[rand_idx[mask]]
                u_embed[mask] = u_mix

                # 混合物品嵌入
                i_mix = lam.view(-1, 1, 1) * i_embed[mask] + (1 - lam.view(-1, 1, 1)) * i_embed[rand_idx[mask]]
                i_embed[mask] = i_mix

        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
        u_v = u_embed.repeat(1, items.shape[1]).view(items.shape[0], items.shape[1], -1)
        i_v = i_embed
        return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v': i_v}


class LGCNEncoder(nn.Module):
    def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3):
        super(LGCNEncoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.emb_size = emb_size
        self.layers = [emb_size] * n_layers
        self.norm_adj = norm_adj

        self.embedding_dict = self._init_model()
        # 不要 .cuda()，不要 if torch.cuda.is_available()
        sparse_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        self.register_buffer('sparse_norm_adj', sparse_adj)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
        })
        return embedding_dict

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        # 1. 先将coo.row和coo.col合并为单个numpy数组，避免列表转tensor的低效
        indices = np.vstack([coo.row, coo.col])  # 形状：(2, N)
        # 2. 使用推荐的torch.sparse_coo_tensor创建稀疏张量
        # 3. 直接指定设备，避免后续再迁移
        sparse_matrix = torch.sparse_coo_tensor(
            indices=torch.from_numpy(indices).long(),  # 转为LongTensor
            values=torch.from_numpy(coo.data).float(),  # 转为FloatTensor
            size=coo.shape,
            # device=device  # 直接在GPU上创建（若可用），避免CPU→GPU迁移开销
        )
        # 若需要保留梯度（根据模型需求），添加requires_grad
        # sparse_matrix.requires_grad = False
        return sparse_matrix

    def forward(self, users, items):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings


class LightGCN(GeneralModel, LightGCNBase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'batch_size', 'mix_prob', 'mix_alpha']

    @staticmethod
    def parse_model_args(parser):
        parser = LightGCNBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = LightGCNBase.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}


class LightGCNImpression(ImpressionModel, LightGCNBase):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'n_layers', 'batch_size', 'mix_prob', 'mix_alpha']

    @staticmethod
    def parse_model_args(parser):
        parser = LightGCNBase.parse_model_args(parser)
        return ImpressionModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ImpressionModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return LightGCNBase.forward(self, feed_dict)
