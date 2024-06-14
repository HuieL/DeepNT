import urllib.request
import torch.nn as nn
import pandas as pd
from sklearn import metrics

# urllib.request.urlretrieve(
#     'https://data.dgl.ai/tutorial/dataset/members.csv', './members.csv')
# urllib.request.urlretrieve(
#     'https://data.dgl.ai/tutorial/dataset/interactions.csv', './interactions.csv')
import torch.nn.functional as F
import numpy as np
members = pd.read_csv('./members.csv')
members.head()

interactions = pd.read_csv('./interactions.csv')
interactions.head()
import pandas as pd
import dgl
from dgl.data import DGLDataset
import torch
import os
MSELoss = nn.MSELoss(reduce=True, size_average=True)
class KarateClubDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='karate_club')

    def process(self):
        nodes_data = pd.read_csv('./members.csv')
   
        edges_data = pd.read_csv('./interactions.csv')
        node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        features = np.load('GCN/raw/features.npy')
        node_features = torch.as_tensor(torch.from_numpy(features), dtype=torch.float32)
        
        node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())  # 将Club属性变为category类型，往往作为label 并且转为0,1
        label_list = np.load('GCN/raw/label_list.npy')
        edge_labels = torch.as_tensor(torch.from_numpy(label_list), dtype=torch.float32)
  
        edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
        print(edges_src)
        print(nodes_data.shape[0])
        edge_index = np.load('GCN/raw/pair.npy')
        edge_index = torch.as_tensor(torch.from_numpy(edge_index), dtype=torch.long)
        edges_src = edge_index[0]
        edges_dst = edge_index[1]
        
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=416)
        self.graph.ndata['feat'] = node_features
        self.graph.edata['label'] = edge_labels
        # self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = 914
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.edata['train_mask'] = train_mask
        self.graph.edata['val_mask'] = val_mask
        self.graph.edata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

dataset = KarateClubDataset()
g = dataset[0]

# print(graph)
from dgl.nn import GraphConv
import torch.nn as nn
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes,edges_src,edges_dst):
        super(GCN, self).__init__()
        self.n =  18
        self.gamma =  int(2*self.n) 
        self.conv1 = GraphConv(in_feats, self.gamma)
        self.conv2 = GraphConv(self.gamma, self.gamma)
        

        self.eta =  int(2*self.n) 

        self.linear1 = nn.Linear(self.gamma*2, self.eta)   
        self.sigmoid = nn.Sigmoid()                 
        self.linear2 = nn.Linear(self.eta, self.eta)    
        self.linear3 = nn.Linear(self.eta, 1,bias=False)    
        self.edges_src = edges_src
        self.edges_dst = edges_dst

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        # print('hahah')
        # print(h.size())


        hidden_1 = self.linear1( torch.cat([h[self.edges_src ],h[self.edges_dst]],1))
        # print(hidden_1.size())
        hidden_1 = self.sigmoid(hidden_1)
       
        hidden_2 = self.linear2(hidden_1)
        hidden_2 = self.sigmoid(hidden_2)

        output = self.linear3(hidden_2)
    
        return output
        # return h

# Create the model with given dimensions
# print(g.ndata['feat'].size())

edge_index = np.load('GCN/raw/pair.npy')
edge_index = torch.as_tensor(torch.from_numpy(edge_index), dtype=torch.long)
edges_src = edge_index[0]
edges_dst = edge_index[1]

model = GCN(g.ndata['feat'].shape[1], 16, 9,edges_src,edges_dst)
def compute_loss(label,output):
    loss = MSELoss(label,output)
    return loss 
def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 10000000
    best_test_acc = 1000000

    features = g.ndata['feat']
    labels = g.edata['label']
    train_mask = g.edata['train_mask']
    val_mask = g.edata['val_mask']
    test_mask = g.edata['test_mask']
    for e in range(1000):
        # Forward
        pred = model(g, features)
        pred = pred.reshape(-1)

        # Compute prediction
        # pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # loss = F.cross_entropy(pred[train_mask], labels[train_mask])
        # print(pred[train_mask].size())
        # print(labels[train_mask].size())
        loss = compute_loss(pred[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        train_acc = metrics.mean_squared_error(pred[train_mask].detach().numpy(), labels[train_mask].detach().numpy())
        # print(MSE)
        val_acc = metrics.mean_squared_error(pred[val_mask].detach().numpy(), labels[val_mask].detach().numpy())
        test_acc = metrics.mean_squared_error(pred[test_mask].detach().numpy(), labels[test_mask].detach().numpy())
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc > val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
# model = GCN(g.ndata['feat'].shape[1], 16, len(g.edata['label'] ))
train(g, model)