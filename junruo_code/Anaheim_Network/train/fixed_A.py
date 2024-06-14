# from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import networkx as nx
import numpy as np
torch.set_printoptions(profile="full")
from sklearn import metrics
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

df = pd.read_csv('shuffled_data_small/small_list_Anaheim_Network.csv')

# print(df)
init_list = df['init_list'].values.tolist()
end_list = df['end_list'].values.tolist()
weight = df['weight'].values.tolist()

node_num = 416

MSELoss = nn.MSELoss(reduce=True, size_average=True)


# node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())  # 将Club属性变为category类型，往往作为label 并且转为0,1

edge_index = np.load('shuffled_data_1000/pair.npy')

edges_src = edge_index[0]
edges_dst = edge_index[1]
# print(len(edges_src))
des = []
src = []
eweight = []
# label_list = []
t = 0
for i in range(node_num):
    for j in range(node_num):
        if t < len(edges_src) and edges_src[t] == i  and edges_dst[t] == j:
            eweight.append(weight[t])
            # labels.append()
            # label_list.append(label[t])
            t = t + 1
        else:
            eweight.append(0)
            # label_list.append(0)
        des.append(j)
        src.append(i)
eweight = np.array(eweight)
eweight = eweight.reshape(node_num,node_num)
# print(type(eweight))
# eweight = np.matrix(eweight)
# eweight = torch.tensor(eweight)
# print(eweight.size())


def normalize(A , symmetric=True):
	# A = A+I
	A = A + torch.eye(A.size(0))
	# 所有节点的度
	d = A.sum(1)
	if symmetric:
		#D = D^-1/2
		D = torch.diag(torch.pow(d , -0.5))
		return D.mm(A).mm(D)
	else :
		# D=D^-1
		D =torch.diag(torch.pow(d,-1))
		return D.mm(A)
class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            # print("Entered")
            w=module.weight.data
            # w = self.normalize(w)
            # print('wwwwwwwwwww')
            # print(w)
            w=w.clamp(0,10000000000) #将参数范围限制到0.5-0.7之间
            module.weight.data=w
    def normalize(self , A , symmetric=True):
        # A = A+I
        I = torch.eye(A.size(0))
        I = I.cuda()
        A = A + I
        # 所有节点的度
        d = A.sum(1)
        print(d)
        if symmetric:
            #D = D^-1/2
            D = torch.diag(torch.pow(d , -0.5))
            return D.mm(A).mm(D)
        else :
            # D=D^-1
            D =torch.diag(torch.pow(d,-1))
            return D.mm(A)

def compute_loss(label,output):
    loss = MSELoss(label,output)
    return loss 

class GCN(nn.Module):

    def __init__(self , A,dim_in , dim_out,edges_src,edges_dst):
        super(GCN,self).__init__()
        # self.n =  dim_in
        # self.gamma =  int(self.n) 
        self.weighted_A = A
        self.fc1 = nn.Linear(dim_in ,dim_out,bias=False)
        # self.fc2 = nn.Linear(dim_in,dim_in//2,bias=False)
        self.fc3 = nn.Linear(dim_out,dim_out,bias=False)
    # variable = Variable(tensor, requires_grad=True)
    # 	self.etensor = nn.Parameter(eweight)
        # print(self.etensor)
        # self.weighted_A = Variable(eweight, requires_grad=True)



        # self.weighted_A = nn.Linear(34,34,bias=False)
        # self.weighted_A.weight = nn.Parameter(self.A)

        # print(dim_in)
        self.eta =  int(dim_out) 
        self.gamma =  int(dim_out) 
        # self.weighted_A = nn.Linear(node_num,node_num,bias=False)
        self.linear1 = nn.Linear(self.gamma*2, self.eta)   
        self.sigmoid = nn.Sigmoid()                 
        self.linear2 = nn.Linear(self.eta, self.eta)    
        self.linear3 = nn.Linear(self.eta, 1,bias=False)    
        self.edges_src = edges_src
        self.edges_dst = edges_dst
        # self.weighted_A = nn.Parameter(self.weighted_A)
        
    def forward(self,X):
        # print('predict_adj')
        # print(self.weighted_A.device)
        X=self.weighted_A.mm(X)
        # print('这里2')
        # print(X.size())
        X = F.relu(self.fc1(X))

  
        X = self.weighted_A.mm(X)
        X = F.relu(self.fc3(X))
        # weight_norm = torch.norm(self.weighted_A.weight,p=1)
    
        hidden_1 = self.linear1( torch.cat([X[self.edges_src ],X[self.edges_dst]],1))
        # print(hidden_1)
      
        hidden_2 = self.sigmoid(hidden_1)
        output = self.linear3(hidden_2)
        weight_norm = 0
        return output,self.weighted_A,weight_norm

        
        # return output,self.weighted_A,weight_norm
    
#获得空手道俱乐部数据
G = nx.karate_club_graph()


A = eweight
#A需要正规化
A_normed = normalize(torch.FloatTensor(A),True)
A_normed = Variable(torch.FloatTensor(A_normed),requires_grad = True) 
# eweight = Variable(torch.FloatTensor(eweight),requires_grad = True)
N = len(A)
X_dim = N
# 没有节点的特征，简单用一个单位矩阵表示所有节点
X = torch.eye(N,X_dim)

features = X


f = np.load('shuffled_data_small/features.npy')
node_features = torch.as_tensor(torch.from_numpy(f), dtype=torch.float32)
features = node_features
# features = torch.tensor(416,416)
X_dim = 9
features = features.cuda()
label_list = np.load('shuffled_data_small/label_list.npy')
path_labels = torch.as_tensor(torch.from_numpy(label_list), dtype=torch.float32)
path_labels = path_labels.cuda()

n_edges = int(len(label_list)*1.0)#分别选30%，60%，100%的数据
n_train = int(n_edges *0.6)
n_val = int(n_edges * 0.2)

path_labels = path_labels[0:n_edges]
path_index = np.load('shuffled_data_small/pair_aug.npy')
path_index = torch.as_tensor(torch.from_numpy(path_index), dtype=torch.long)
# print(path_index.size())
a_path_src = path_index[0][0:n_edges]
a_path_dst = path_index[1][0:n_edges]

A_normed = A_normed.cuda()

# 我们的GCN模型
model = GCN(A_normed ,X_dim,64,a_path_src,a_path_dst)
#选择adam优化器
# gd = torch.optim.Adam(model.parameters())
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
best_val_acc = 10000000
best_test_acc = 1000000


train_mask = torch.zeros(n_edges, dtype=torch.bool)
val_mask = torch.zeros(n_edges, dtype=torch.bool)
test_mask = torch.zeros(n_edges, dtype=torch.bool)
train_mask[:n_train] = True
val_mask[n_train:n_train + n_val] = True
test_mask[n_train + n_val:] = True
# features = g.ndata['feat']
true_adj = np.array(eweight)
true_adj = torch.from_numpy(true_adj)
	
# print('true_adj')

# print(true_adj)

# print('true_measurement')

# print(path_labels[train_mask])
constraints=weightConstraint()
for e in range(5000):
	#转换到概率空间
    pred,adj,weight_norm = model( features)
    pred = pred.reshape(-1)
    # print(adj)
    # print('edge_weight')
    # print(edge_weight)
    # print(pred.size())
    # distance_adj = F.pairwise_distance(A_normed.reshape(1,-1), adj.reshape(1,-1) )
    # print('label')
    # print(path_labels[train_mask])

    pred = pred.reshape(-1)
    # print('pred')
    # print(pred)



    loss = compute_loss(pred[train_mask], path_labels[train_mask])+10e-4 * weight_norm

    # Compute accuracy on training/validation/test
    # train_acc = (pred[train_mask] == path_labels[train_mask]).float().mean()
    # train_acc = metrics.mean_squared_error(pred[train_mask].detach().numpy(), path_labels[train_mask].detach().numpy())
    train_acc = metrics.mean_absolute_percentage_error(pred[train_mask].detach().cpu().numpy(), path_labels[train_mask].detach().cpu().numpy())
    # print(MSE)
    val_acc = metrics.mean_absolute_percentage_error(pred[val_mask].detach().cpu().numpy(), path_labels[val_mask].detach().cpu().numpy())
    test_acc = metrics.mean_absolute_percentage_error(pred[test_mask].detach().cpu().numpy(), path_labels[test_mask].detach().cpu().numpy())
    # print(test_acc)
    # print(test_acc)
    # Save the best validation accuracy and the corresponding test accuracy.
    if best_val_acc > val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    # Backward
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    # model._modules['weighted_A'].apply(constraints)
    print('predict_measurement')
    print(pred)
    if e % 100 == 0 and e > 1000:
        # print('predict_measurement')
        # print(pred)
        

        print('In epoch {}, loss: {:.3f}, train  mape: {:.3f} , val mape: {:.3f} (best {:.3f}), test mape: {:.3f} (best {:.3f})'.format( e, loss, val_acc, best_val_acc,train_acc, test_acc, best_test_acc))
