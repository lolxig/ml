import numpy as np
import pandas as pd
import time


for i in range(10):
    print(i)

#
# name = np.array(
#     ['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
#      'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
#      'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
#      'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
#      'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
#      'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
#      'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
#      'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio',
#      'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
#      'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
#      'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
#      'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label',
#      ])
# a = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0])
# b = np.argwhere(a == 1)
#
# print(b)
# print(name[b])

# s_from_list = pd.Series([1, 2, 3, 4, 5])
# print(s_from_list)

# s_from_list2 = pd.Series(['aa', 'bb', 'cc', 'dd', 'ee'])
# print(s_from_list2)

# s_from_list = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
# print(s_from_list)

# s_from_dict = pd.Series({'name': '张炎', 'age': 18, 'gender': '男', 'hobby': '编程'})
# print(s_from_dict)

# s_from_list1 = pd.Series([1, 2, 3, 4, 5])
# val1 = s_from_list1[2]
# print(val1)

# s_from_list2 = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
# val2 = s_from_list2['D']
# print(val2)
# val3 = s_from_list2[3]
# print(val3)

# s_from_list1 = pd.Series([1, 2, 3, 4, 5])
# val4 = s_from_list1[2:4]
# print(val4)
# s_from_list2 = pd.Series([1, 2, 3, 4, 5], index=['A', 'A', 'A', 'D', 'E'])
# val5 = s_from_list2['A']
# print(val5)

# s_from_list1 = pd.Series([1, 2, 3, 4, 5])
# val6 = s_from_list1[[1, 3, 4]]
# print(val6)
# s_from_list2 = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
# val6 = s_from_list2[['A', 'C', 'D']]
# print(val6)

# s_from_list1 = pd.Series([1, 2, 3, 4, 5])
# add_s = pd.Series(6)
# new_s = s_from_list1.append(add_s)
# print(s_from_list1)
# print(new_s)
# print(new_s.index)

# s_from_list1 = pd.Series([1, 2, 3, 4, 5])
# add_s = pd.Series(6, index=[7])
# new_s = s_from_list1.append(add_s)
# print(s_from_list1)
# print(new_s)
# print(new_s.index)

# s_from_list1 = pd.Series([1, 2, 3, 4, 5])
# droped = s_from_list1.drop(2)
# print(s_from_list1)
# print(droped)
# s_from_list2 = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
# droped = s_from_list2.drop('E')
# print(s_from_list2)
# print(droped)

# s_from_list2 = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
# print(s_from_list2)
# s_from_list2[3] = 101
# print(s_from_list2)

# s_from_list2 = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
# print(s_from_list2)
# s_from_list2[::2] = 101
# print(s_from_list2)

# s_from_list1 = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
# s_from_list2 = pd.Series(['aa', 'bb', 'cc', 'dd', 'ee'])
# s_from_list3 = pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])
# s_from_list4 = pd.Series([1, 2, 3, 4, 5], index=range(1, 6))
# print(s_from_list1.index)
# print(s_from_list2.index)
# print(s_from_list3.index)
# print(s_from_list4.index)


# input_path = 'C:\\Users\\Administrator\\Desktop\\cicids2018\\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv'
# output_path = 'C:\\Users\\Administrator\\Desktop\\out.bcp'
#
# # 读取数据文件并进行分离
# # ids_data = pd.read_csv(input_path, encoding='utf-8').drop(columns='Timestamp')
# # ids_data_Benign = ids_data[ids_data.loc[:, 'Label'] == 'Benign'].drop(columns='Label')
# # ids_data_Infilteration = ids_data[ids_data.loc[:, 'Label'] == 'Infilteration'].drop(columns='Label')
#
# population = np.zeros((200, 78), dtype=np.int)
# for i in range(200):  # 定义种群的个体数为 n
#     a = np.zeros(78 - 40, dtype=np.int)  # 生成未被选择的特征
#     b = np.ones(40, dtype=np.int)  # 将选择的d维特征定义为个体c中的1
#     c = np.append(a, b)
#     c = (np.random.permutation(c.T)).T  # 随机生成一个d维的个体
#     population[i] = c  # 初代的种群为 population，共有n个个体



# # 取得各个样本的数量和总体样本的数量
# len_total = ids_data.__len__()
# len_Benign = ids_data_Benign.__len__()
# len_Infilteration = ids_data_Infilteration.__len__()

# s111 = ids_data[0]
# s222 = ids_data['Label']
# s222 = ids_data.iloc[:, [0, 1, 2]]
# print(s222)
# print(type(s222))

# df = pd.read_csv(input_path, encoding='utf-8')
# # title = list(df.columns.values)
# # data = np.array(df.loc[:, :])
# #
# # feature_names = df.columns
# #
# # print(type(df))
# # print(type(feature_names))
#
# data = df[df.loc[:, 'Label'] == 'Infilteration']
#
#
# with open('C:\\Users\\Administrator\\Desktop\\out', 'a') as f:
#     for row in data:
#         outStr = ''
#         for elem in row:
#             outStr += elem + '\t'
#         outStr = outStr[:len(outStr) - 1]
#         outStr += '\n'
#         f.write(outStr)


# buff_size = 512 * 1024
# start = time.time()
# with open(input_path, 'rb', buffering=buff_size) as f:
#     with open(output_path, 'wb', buffering=buff_size) as o:
#         s = ''
#         for line in f:
#             #     # print(line)
#             #     # s = str(line[:-1], encoding='utf-8')
#             #     s += line
#             #     if len(s) >= buff_size:
#             #         o.write(s)
#             #         s = b''
#             #     # pass
#             # o.write(s)
#             #     s = str(line[:-1], encoding='utf-8').split()
#
#             o.write(line)
# print(time.time() - start)
