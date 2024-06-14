import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_scatter import scatter_add
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool, Linear

from typing import Callable, Union
from torch_geometric.typing import OptPairTensor



class PathNN(nn.Module):
    """
    Path Neural Networks that operate on collections of paths. Uses 1 LSTM shared across convolutional layers.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim,
            cutoff,
            n_classes,
            dropout,
            device,
            residuals=True,
            encode_distances=False,
            l2_norm=False,
            predict=True,
    ):
        super(PathNN, self).__init__()
        self.cutoff = cutoff
        self.device = device
        self.residuals = residuals
        self.dropout = dropout
        self.encode_distances = encode_distances
        self.l2_norm = l2_norm
        self.predict = predict

        # Feature Encoder that projects initial node representation to d-dim space
        self.feature_encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        conv_class = PathConv

        # 1 shared LSTM across layers
        if encode_distances:
            self.distance_encoder = nn.Embedding(cutoff, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim * 2,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=True,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=True,
            )

        self.convs = nn.ModuleList([])
        for _ in range(self.cutoff - 1):
            self.convs.append(
                conv_class(
                    hidden_dim,
                    self.lstm,
                    batch_norm=nn.Identity(),
                    residuals=self.residuals,
                    dropout=self.dropout,
                )
            )

        self.hidden_dim = hidden_dim
        if self.predict:
            self.linear1 = Linear(hidden_dim, hidden_dim)
            self.linear2 = Linear(hidden_dim, n_classes)

        self.reset_parameters()

    def reset_parameters(self):

        for c in self.feature_encoder.children():
            if hasattr(c, "reset_parameters"):
                c.reset_parameters()
        self.lstm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.predict:
            self.linear1.reset_parameters()
            self.linear2.reset_parameters()
        if hasattr(self, "distance_encoder"):
            nn.init.xavier_uniform_(self.distance_encoder.weight.data)

    def forward(self, data):

        # Projecting init node repr to d-dim space
        h = self.feature_encoder(data.x)

        # Looping over layers
        for i in range(self.cutoff - 1):
            if self.encode_distances:
                # distance encoding with shared distance embedding
                dist_emb = self.distance_encoder(getattr(data, f"sp_dists_{i+2}"))
            else:
                dist_emb = None

            # Euclidean normalization
            if self.l2_norm:
                h = F.normalize(h, p=2, dim=1)

            h = self.convs[i](h, getattr(data, f"path_{i+2}"), dist_emb)

        return h


class PathConv(nn.Module):
    r"""
    The Path Aggregator module that computes result of Equation 2.
    """

    def __init__(
            self, hidden_dim, rnn: Callable, batch_norm: Callable, residuals=True, dropout=0
    ):
        super(PathConv, self).__init__()
        self.rnn = rnn
        self.bn = batch_norm
        self.hidden_dim = hidden_dim
        self.residuals = residuals
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.bn, "reset_parameters"):
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], paths, dist_emb=None):

        h = x[paths]

        # Add distance encoding if needed
        if dist_emb is not None:
            h = torch.cat([h, dist_emb], dim=-1)

        _, (h, _) = self.rnn(h)

        # Summing paths representations based on starting node
        h = scatter_add(
            h.squeeze(0),
            paths[:, -1],
            dim=0,
            out=torch.zeros(x.size(0), self.hidden_dim, device=x.device),
        )

        # Residual connection
        if self.residuals:
            h = self.bn(h + x)
        else:
            h = self.bn(h)

        # ReLU non linearity as the phi function
        h = F.relu(h)

        return h

class GCN(nn.Module):
    def __init__(self, A, dim_in, hidden_dims, dim_out):
        super(GCN, self).__init__()
        self.A_hat = A + torch.eye(A.size(0))
        D_hat_diag = torch.sum(self.A_hat, dim=1).pow(-0.5)
        self.D_hat = torch.diag(D_hat_diag)

        self.gc1 = nn.Linear(dim_in, hidden_dims[0], bias=False)
        self.gc2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
        self.gc3 = nn.Linear(hidden_dims[1], dim_out, bias=False)

    def forward(self, X):
        A_hat = torch.mm(torch.mm(self.D_hat, self.A_hat), self.D_hat)
        X = F.relu(self.gc1(torch.mm(A_hat, X)))
        X = F.relu(self.gc2(torch.mm(A_hat, X)))
        X = self.gc3(torch.mm(A_hat, X))
        return X

class NTGCN(nn.Module):
    def __init__(self, A_init, dim_in, hidden_dims, dim_out):
        super(NTGCN, self).__init__()
        self.A = nn.Parameter(A_init + torch.eye(A_init.size(0)))

        self.gc1 = nn.Linear(dim_in, hidden_dims[0], bias=False)
        self.gc2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
        self.gc3 = nn.Linear(hidden_dims[1], dim_out, bias=False)

    def forward(self, X):
        D_hat_diag = torch.sum(self.A, dim=1).pow(-0.5)
        D_hat = torch.diag(D_hat_diag)
        A_hat = torch.mm(torch.mm(D_hat, self.A), D_hat)
        X = F.relu(self.gc1(torch.mm(A_hat, X)))
        X = F.relu(self.gc2(torch.mm(A_hat, X)))
        X = self.gc3(torch.mm(A_hat, X))
        return X


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, 0.6, training=self.training)
        h_prime = torch.matmul(attention, h)

        return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid) for _ in range(3)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid, nout)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
