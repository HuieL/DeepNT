from code import interact
import re
import torch
import torch.nn as nn
# import torch.nn.Module
from torch import optim
import pandas as pd
import random
import torch.utils.data as data
import time 
from sklearn import metrics
import numpy as np

MSELoss = nn.MSELoss(reduce=True, size_average=True)
class NeuTomography(nn.Module):

    # 初始化网络模型
    def __init__(self):
        super(NeuTomography, self).__init__()
        self.n =  18 ########替换输入
        self.gamma =  int(0.3*self.n)  ########替换输入

        self.linear1 = nn.Linear(self.n, self.gamma)   
        self.sigmoid = nn.Sigmoid()                 
        self.linear2 = nn.Linear(self.gamma, self.gamma)    
        self.linear3 = nn.Linear(self.gamma, self.gamma)    
      
        self.linear4 = nn.Linear(self.gamma, 1,bias=False)    
        # self.sigmoid = nn.Sigmoid(dim=1)     
        # for m in self.children():
            # if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform_(self.linear1.weight, gain=1)
        # torch.nn.init.xavier_uniform_(self.linear1.bias, gain=1)
        # torch.nn.init.xavier_uniform_(self.linear2.weight, gain=1)
        # torch.nn.init.xavier_uniform_(self.linear2.bias, gain=1)
        # torch.nn.init.xavier_uniform_(self.linear3.weight, gain=1)
        # torch.nn.init.xavier_uniform_(self.linear3.bias, gain=1)
        # torch.nn.init.xavier_uniform_(self.linear4.weight, gain=1)
        # torch.nn.init.xavier_uniform_(self.linear4.bias, gain=1)
                # torch.nn.init.normal_(m.weight, mean=0, std=1)
                # torch.nn.init.normal_(m.bias, mean=0, std=1)
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)


    # 前向传播
    def forward(self, x):
        
        hidden_1 = self.linear1(x)
        hidden_1 = self.sigmoid(hidden_1)
        # print('layer1')
        # print(hidden_1)
     
        hidden_2 = self.linear2(hidden_1)
        hidden_2 = self.sigmoid(hidden_2)
        # print('layer2')

        # print(hidden_2)

        hidden_3 = self.linear3(hidden_2)
        hidden_3 = self.sigmoid(hidden_3)
        # print('layer3')

        # print(hidden_3)



        output = self.linear4(hidden_3)
        # print(output)
        
       
        return output




      
        
  

def compute_loss(label,output):
    loss = MSELoss(label,output)
    return loss
def load_all(train= True):
    # f = open('augment_and_test.csv','rb')
    #     print(f)

    ini_term_pair = np.load('mlp/ini_term_pair.npy')
    # ini_term_pair = torch.from_numpy(ini_term_pair) 
    # ini_term_pair = torch.tensor(ini_term_pair, dtype=torch.float)
    ini_term_pair = torch.as_tensor(torch.from_numpy(ini_term_pair), dtype=torch.float32)

  

    label_list = np.load('mlp/label_list.npy')
    # label_list = torch.from_numpy(label_list)
    # label_list = torch.tensor(label_list, dtype=torch.float)
    label_list = torch.as_tensor(torch.from_numpy(label_list), dtype=torch.float32)

    
    # print(len(label_list))

    test_ini_term_pair = np.load('mlp/test_ini_term_pair.npy')
    # test_ini_term_pair = torch.from_numpy(test_ini_term_pair)
    # test_ini_term_pair = torch.tensor(test_ini_term_pair, dtype=torch.float)
    test_ini_term_pair = torch.as_tensor(torch.from_numpy(test_ini_term_pair), dtype=torch.float32)

    # print(test_ini_term_pair.size())


    test_label = np.load('mlp/test_label.npy')
    test_label = torch.as_tensor(torch.from_numpy(test_label), dtype=torch.float32)

    # test_label = torch.from_numpy(test_label)
    # test_label = torch.tensor(test_label, dtype=torch.float)
    # print(test_label.size())

    return ini_term_pair, label_list,test_ini_term_pair,test_label

class PATData(data.Dataset):
    def __init__(self, ini_term_pair, 
				label, num_ng=0, is_training=None):
            super(PATData, self).__init__()
            self.pairs = ini_term_pair
            self.label = label
    def __len__(self):
        return len(self.pairs) 
   
    def __getitem__(self, idx):
        # print(self.pairs[idx])
        # print(self.label[idx])
        return self.pairs[idx], self.label[idx]



# path = '/Users/gaojunruo/Desktop/tomography/network_tomography-main/paths_direct'
ini_term_node_train, label_train ,ini_term_node_test, label_test = load_all()
num_ng = 10
batch_size = 128
train_dataset = PATData(
		ini_term_node_train, label_train, num_ng, True)
test_dataset = PATData(
		ini_term_node_test, label_test, num_ng,  False)
train_loader = data.DataLoader(train_dataset,
		batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=batch_size, shuffle=False, num_workers=0)


learning_rate = 10e-3
model = NeuTomography()
model = model.cuda()
epochs = 1000
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
beta = 0.6
for ep in range(epochs):
    model.train() 
    start_time = time.time()
    l = 0 

    i = 0 
    for ini_term_node_train, label_train in train_loader:
        # print(ini_term_node_train.size())
        # print(label_train)

        ini_term_node_train = ini_term_node_train.cuda()

        label_train = label_train.cuda()
        # optimizer.zero_grad()
        output = model(ini_term_node_train)
        # print(output)
        # print('label')
        # print(label_train)
        
        output = output.reshape(-1)
        print('predict')
        print(output)
        print('label')
        print(label_train)
        loss = compute_loss(label_train, output)
        # print('train')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l = l + loss.item()
        i = i + 1
    
    print(" epoch {:03d}: training mse loss={:.3f}".format(ep,l/i))
    
    model.eval()
    mse = []
    for  ini_term_node_test, label_test in test_loader:
        ini_term_node_test = ini_term_node_test.cuda()
        label_test = label_test.cuda()
        output = model(ini_term_node_test)

        output = output.reshape(-1)
 
        MSE = metrics.mean_squared_error(label_test.cpu().detach().numpy(), output.cpu().detach().numpy())
        update = output * (1-beta) + label_test * beta
        label_test = update 


        mse.append(MSE)
    print("epoch {:03d}: test mse = {:.3f}".format(ep, np.mean(mse)))

