from imp import source_from_cache
import networkx as nx   #导入networkx包
import random			#导入random包
import matplotlib.pyplot as plt #导入画图工具包
import numpy as np
from collections import defaultdict

# /Users/gaojunruo/Desktop/tomography/network_tomography-main/Austin_Network.xlsx


import openpyxl
 


import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
import math
# import xlwt
# import xlsxwriter

import openpyxl
 

node_num = 416

file = "Anaheim_Network"


workbook=openpyxl.load_workbook(file+".xlsx")
worksheet=workbook.worksheets[0]

init_list = []
end_list = []
for index,col in enumerate(worksheet.rows):
    # print(index)
    # print(col)
    if index==0:
        continue
    else:
        # col[4].value=col[2].value/col[3].value #手动在xlsx加velocity
        col[0].value=col[0].value-1
        col[1].value=col[1].value-1

#枚举出来是tuple类型，从0开始计数
 
workbook.save(filename="new_"+file+".xlsx")



workbook=openpyxl.load_workbook("new_"+file+".xlsx")
worksheet=workbook.worksheets[0]

init_list = []
end_list = []
weight = []
for index,col in enumerate(worksheet.rows):
    # print(index)
    # print(col)
    if index==0:
        continue
    else:
        init_list.append(col[0].value)
        end_list.append(col[1].value)
        weight.append(col[2].value)

# print('hhh')

# print(len(set(init_list)))

# print(len(init_list))
# print(len(end_list))
# print(len(weight))

frame = pd.DataFrame({'init_list': init_list,'end_list':end_list,'weight': weight})
print(frame)
frame.to_csv("list_"+file+".csv")

for i in range(node_num):
    if i in init_list:
        continue
    else:
        print('wrong')
        print(i)




pair = []
import pandas as pd


for i in range(len(init_list)):  


    pair.append([init_list[i],end_list[i],weight[i]])
 



import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
import math
# import xlwt
# import xlsxwriter
from collections import defaultdict

ini_term_dict = defaultdict(list)



for i in range(len(init_list)):
    # if ini_node[i] in ini_term_dict:
        ini_term_dict[init_list[i]].append([end_list[i],weight[i]])
lst = []
ends = []
for i in ini_term_dict:
    lst.append(i)
    ends.append(ini_term_dict[i])

# print(len(lst))
# print(len(ends))



inf = 10e9

class graphMatrix:
    def __init__(self):
        self.dict={}  # 结点对应矩阵下标的字典
        self.length=0 # 矩阵的长度
        self.matrix=[]

    # 添加节点
    def addVertex(self,key):
        self.dict[key]=self.length # 键(顶点名称)->值(在矩阵中的下标)
        self.dict[self.length]=key # 键(在矩阵中的下标)->值(顶点名称)


        lst=[]
        for i in range(self.length):  # 创建一行长度为self.length,全是0的列表
            lst.append(inf)
        self.matrix.append(lst)

        for row in self.matrix:  # 为矩阵中每行添加新的一个单元，单元中值为0
            row.append(inf)

        self.length += 1
        print(self.length)

    # 添加无向边
    def addNoDirectLine(self,start,ends): #起始字符，终点(字符+权值)数组
        startIndex=self.dict[start] # 起点在二维矩阵中的下标
        for row in ends:  # 遍历终点数组
            endKey=row[0]
            endIndex=self.dict[endKey]  # 终点在二维矩阵中的下标
            weight=row[1]
            # print(startIndex)
            # print(startIndex)
            self.matrix[startIndex][endIndex]=weight  # 权值赋值
            self.matrix[endIndex][startIndex]=weight

    # 添加有向边
    def addDirectLine(self,start,ends):
        startIndex=self.dict[start] # 起点在二维矩阵中的下标
        # print(ends)
        # print(self.dict)
        for row in ends:  # 遍历终点数组
            endKey=row[0]
            endIndex=self.dict[endKey]  # 终点在二维矩阵中的下标
            weight=row[1]
            self.matrix[startIndex][endIndex]=weight  # 权值赋值

def construct_matrix():

    # lst=['北京','天津','郑州','青岛']
    # ends = [[['天津',138],['郑州',689]],[['郑州',700],['青岛',597]],[['青岛',725]]]
    print('??')
    print(len(lst))
    gm=graphMatrix()
    for k in lst:
        gm.addVertex(k)
    for i in range(len(ends)):
        gm.addNoDirectLine(lst[i],ends[i])

    all_matrices = []
    print('\n')
    for i in range(len(gm.matrix)):
        all_matrices.append(gm.matrix[i])
        # print(gm.matrix[i])
    # print(all_matrices)
    return all_matrices


# construct_matrix()


######################dijistra求最短路径，边权重可以是长度和时间。

import numpy as np
def startwith(start: int, mgraph: list) -> list:
# 存储已知最小长度的节点编号 即是顺序
    passed = [start]
    nopass = [x for x in range(len(mgraph)) if x != start]

    dis = mgraph[start]
    # print(dis)
    # 创建字典 为直接与start节点相邻的节点初始化路径
    dict_ = {}
    dict_weight = {}
    for i in range(len(dis)):
        if dis[i] != np.inf:
            dict_[str(i)] = [start]
            dict_weight[str(i)] = [0]

    while len(nopass):
        idx = nopass[0]
        for i in nopass:
            if dis[i] < dis[idx]: 
                idx = i

        nopass.remove(idx)
        passed.append(idx)

        for i in nopass:
            if dis[idx] + mgraph[idx][i] < dis[i]:
                dis[i] = dis[idx] + mgraph[idx][i]
                dict_[str(i)] = dict_[str(idx)] + [idx] #如果之前经历过idx,而且当前到i的距离最短
                dict_weight[str(i)] =  dict_weight[str(idx)] + [dis[idx] + mgraph[idx][i]]

    return dis,dict_,dict_weight

import pandas as pd
import csv


if __name__ == "__main__":

    matrix = construct_matrix()
 
    matrix = np.array(matrix)
    row, col = np.diag_indices_from(matrix)
    matrix[row,col] = 0
    matrix = matrix.tolist()

    term_node_list = []
    ini_node_list = []
    metric = []
    for j in range(node_num):
        start = j

        dis,dict_,dict_weight = startwith(start, matrix)#从0开始的path。
        ini_node = [j]*node_num
        term_node = list(range(node_num))#0-415
        ini_node_list = ini_node_list + ini_node
        term_node_list = term_node_list + term_node
        metric = metric + dis
 
    ini = []
    end = []
    label = []
    # print(type(ini_node_list))
    for i in range(len(ini_node_list)):
       
            if ini_node_list[i] ==  term_node_list[i]:
                continue
            else:
                ini.append(ini_node_list[i])
                end.append(term_node_list[i])
                label.append(metric[i])


    data = {
        'init_node': ini,
    'term_node': end,
    'metric': label
    }
    df = pd.DataFrame(data)
    # print(df['init_node'].values)
    # df['init_node'].values
    df.to_csv(file+"_augment_and_test.csv",index=False,sep=',')
