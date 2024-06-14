import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        #self.l4=nn.Linear(torch.tensor([[2.,2.,2.],[2.,2.,2.]])) 2*3 matrix
        self.l4=nn.Linear(3,2)

        self.l1=nn.Linear(2,50)
        self.l2=nn.Linear(50,10)
        self.l3=nn.Linear(10,1)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        x=self.l4(x)
        x=self.l1(x)
        x=self.l2(x)
        x=self.l3(x)
        x=self.sig(x)
        # print(self.l3)
        #print(self.l4)
        return(x)
class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            # print("Entered")
            w=module.weight.data
            w=w.clamp(0,0.2)
            module.weight.data=w
# Applying the constraints to only the last layer
constraints=weightConstraint()
model=Model()
# print(model._modules['l3'].weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(0,100):
    #print('before optimizing:')
    #print(model._modules['l4'].weight)
    x = torch.randn(2, 3, requires_grad=True)
    y = model(x)
    loss = torch.sum(y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('before applying constraint:')
    print(model._modules['l4'].weight)
    model._modules['l4'].apply(constraints)
    print('after applying constraint:')
    print(model._modules['l4'].weight)
# print(model.l4)