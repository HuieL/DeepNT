import numpy as np
import random
import pandas as pd
import random

# init = [0, 0, 0, 0, 0, 0, 0, 0, 0 , 0 , 0 , 0 , 0 , 0 , 1 ,1,  1,  1, 2, 2, 2, 2,  2,  2,  2, 3,  3, 4,  4, 5,  5,  6, 7,  8,  8,  8,  9, 10, 11, 12, 13, 13, 14, 14, 15, 15, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 33]
# end =  [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 17, 19, 21, 31, 2, 3, 19, 30, 3, 7, 8, 9, 13, 28, 32, 7, 13, 6, 10, 6, 10, 16, 1, 30, 32, 33, 33, 0,  0, 3, 1, 33, 32, 33, 32, 33, 5, 1, 32, 33, 33, 32, 33, 1, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 2, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 26]
# label =   [4, 13, 12, 6, 16, 14, 12, 3, 7, 10, 18, 18, 19, 14, 9, 2, 17, 12, 9, 11, 18, 16, 17, 8, 19, 13, 16, 6, 12, 8, 16, 15, 6, 7, 15, 11, 18, 7, 17, 12, 10, 18, 7, 16, 19, 18, 13, 14, 2, 3, 5, 18, 5, 9, 7, 3, 13, 2, 12, 16, 17, 1, 1, 15, 1, 16, 14, 18, 2, 16, 11, 8, 18, 14, 1, 3, 14, 7]

# init = [0 , 0 , 0 , 1, 1 , 2 , 2 , 3, 3 , 3, 4, 4, 4 , 5 , 6 , 7 , 8 , 9]
# end = [1 , 4 , 9 , 2, 6 , 5 , 6 , 4, 6 , 7, 5, 6, 9 , 6 , 7 , 4 , 7 , 1]
# label =  [19, 11, 17, 6, 15, 10, 14, 1, 11, 8, 6, 9, 17, 17, 15, 15, 10, 19]


df = pd.read_csv('list_Anaheim_Network.csv')

print(df)
init = df['init_list'].values.tolist()
end = df['end_list'].values.tolist()
label = df['weight'].values.tolist()






label_list = []
for i in range(len(label)):
        label_list.append(float(label[i]))
# print(len(init))
# print(len(label))
exit_node = set(init+end)
print(len(exit_node)==max(exit_node)+1) #表示数字是连续的


df_aug = pd.read_csv('Anaheim_Network_augment_and_test.csv',names=['init_node','term_node','metric'],sep=',') 


init_aug  = df_aug['init_node'].values
end_aug  = df_aug['term_node'].values
label  = df_aug['metric'].values
init_aug_list = []
end_aug_list = []
# label_aug = []
all_pairs = []
for i in range(1,len(label)):
    # if i == 0:
    #     label_aug.append(label[i])
    # else:
        # label_list.append(float(label[i]))
        # init_aug_list.append(int(init_aug[i]))
        # end_aug_list.append(int(end_aug[i]))
        all_pairs.append((int(init_aug[i]),int(end_aug[i]) ,float(label[i])))
# ini_end_label = list(zip(init_aug,end_aug,label_aug))
# init_index = list(range(len(ini_end_label)))[1:]
# slice = random.sample(init_index, int(len(init_index)*0.3) )
# test_index = list(set(init_index)-set(slice))

random.shuffle(all_pairs) 
for i in range(len(all_pairs)):
    # if i == 0:
    #     label_aug.append(label[i])
    # else:
        label_list.append(all_pairs[i][2])
        init_aug_list.append(all_pairs[i][0])
        end_aug_list.append(all_pairs[i][1])
print(init_aug_list)
features = []

for i in range(len(exit_node)):
    feature = [ i  >>d & 1 for d in range(9)][::-1] #第*个点，的二进制
    features.append(feature)
    
    # init_node = init[i]
    # end_node = end[i]
    # v = init_node
    # init_list =[ v  >>d & 1 for d in range(9)][::-1] #record[1] = [0,0,0,0,0,0,0,0,0]
    # v = end_node
    # term_list =[ v  >>d & 1 for d in range(9)][::-1] #record[1] = [0,0,0,0,0,0,0,0,0]
    # ini_term_pair.append(init_list+term_list)

# # print(2)
# #补充数据集
aug_data = []


# print(ini_term_pair)
# print(label_list)
node_pair_aug = []
node_pair_aug.append(init+init_aug_list)
node_pair_aug.append(end+end_aug_list)



node_pair = []
node_pair.append(init)
node_pair.append(end)
# print(pair)
# print(len(features))
# print(len(node_pair[0]))
print(features)
np.save('shuffled_data/features.npy',features) # 保存为.npy格式
np.save('shuffled_data/pair_aug.npy',node_pair_aug) # 保存为.npy格式
np.save('shuffled_data/pair.npy',node_pair) # 保存为.npy格式
np.save('shuffled_data/label_list.npy',label_list) # 保存为.npy格式
