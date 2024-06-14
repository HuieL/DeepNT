import pandas as pd
import numpy as np
import networkx as nx
import random
import os
from itertools import islice
from torch.utils.data import Dataset
import random
import scipy.sparse as sp
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random
from sklearn.model_selection import train_test_split

def load_data(dir):
    G = nx.Graph()
    for filename in os.listdir(dir):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(dir, filename)
            df = pd.read_excel(filepath, skiprows=2)
            edges = df[['init_node', 'term_node']].values
            for edge in edges:
                weight = random.randint(1, 10)
                G.add_edge(edge[0], edge[1], weight=weight)
    return G

def generate_path(G, source, target, mode='sum'):
    if mode == 'sum':
        path = nx.shortest_path(G, source=source, target=target, weight='weight')
        path_value = sum(nx.path_weight(G, path, 'weight'))
    elif mode == 'multiplex':
        for u, v, d in G.edges(data=True):
            G[u][v]['log_weight'] = np.log(d['weight'])
        path = nx.shortest_path(G, source=source, target=target, weight='log_weight')
        path_value = np.exp(sum(nx.path_weight(G, path, 'log_weight')))
    elif mode == 'max':
        all_paths = nx.all_simple_paths(G, source=source, target=target)
        max_weight = 0
        for p in all_paths:
            current_max = max([G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)])
            if current_max > max_weight:
                max_weight = current_max
                path = p
        path_value = max_weight
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    return path, path_value


class NTDataset(Dataset):
    def __init__(self, G, rate, mode='sum', train_ratio=0.8):
        self.G = G
        self.mode = mode
        self.nodes = list(G.nodes)
        self.monitor_nodes = random.sample(self.nodes, int(len(self.nodes) * rate))
        self.samples = []
        self.train_samples = []
        self.test_samples = []

        for monitor_node in self.monitor_nodes:
            target_nodes = [node for node in self.nodes if node != monitor_node]
            # 确保每个节点都有机会被作为目标节点
            random.shuffle(target_nodes)
            for target_node in target_nodes[:int(len(target_nodes) * 0.1)]:  # 仅为20%的节点生成样本
                path, path_value = generate_path(self.G, monitor_node, target_node, self.mode)
                self.samples.append((monitor_node, target_node, path_value))

        # 分割训练集和测试集
        random.shuffle(self.samples)
        train_size = int(len(self.samples) * train_ratio)
        self.train_samples = self.samples[:train_size]
        self.test_samples = self.samples[train_size:]

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        return self.train_samples[idx]

    def get_test_samples(self):
        return self.test_samples


class S2VGraph(object):
    def __init__(self, g, label=None, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0

        self.max_neighbor = 0


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor( # torch.sparse
        torch.LongTensor(indices),
        torch.FloatTensor(coo.data),
        coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)


def to_torch(X):
    if sp.issparse(X):
        X = to_nparray(X)
    return torch.FloatTensor(X)

def to_nparray(X):
    if sp.isspmatrix(X):
        return X.toarray()
    else: return X

def sp2adj_lists(X):
    assert sp.isspmatrix(X), 'X should be sp.sparse'
    adj_lists = []
    if sp.isspmatrix(X):
        for i in range(X.shape[0]):
            neighs = list( X[i,:].nonzero()[1] )
            adj_lists.append(neighs)
        return adj_lists
    else:
        return None




