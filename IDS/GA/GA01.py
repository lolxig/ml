# 针对IDS的GA算法
# ver0.01

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 输入数据文件路径
input_path = "C:\\Users\\Administrator\\Desktop\\cicids2018\\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"

# 初始参数
init_pop = 200  # 初始种群大小，目前选择200
ites = 50  # 迭代次数，目前选为50
nr_select = 40  # 需要选择出来的特征数量
reject_list = ['Timestamp']  # 删除不用的特征

# 类数据分离，并进行均值计算
class_lables = set()  # 类别标签列表
datasets = set()  # 数据集字典，按照[类别标签-数据矩阵]的形式排列
mean = set()  # 均值字典，按照[类别标签-均值向量]的形式排列
total_mean = 0  # 整体均值


# 读入数据并进行预处理，删除不必要的字段，将数据拆分成特征集和标签集两个部分
#
# 输入参数：
#   - input_path_t 输入文件路径
#
# 返回值：
#   - title 每个字段的名字
#   - features 数据的特征集
#   - lables 数据的标签值
def read_data():
    # 读取数据
    df = pd.read_csv(input_path, encoding='utf-8')
    title = list(df.columns.values)
    data = np.array(df.loc[:, :])

    # 读取之后根据实际需求进行预处理
    # 这里需要将时间戳列直接删掉，将读入的数据拆分成特征集和标签集两个部分
    # 其中时间戳是第3列数据(从1开始计数)，标签是最后一列数据
    # 删除时间戳
    title.pop(2)
    title.pop(-1)
    data = np.delete(data, 2, 1)

    # 分离特征集和标签集
    features = data[:, :len(title) - 1]
    lables = data[:, -1]

    # 获取类别标签列表
    for elem in lables:
        class_lables.add(elem)

    # 根据类别标签列表分离不同类的数据，并计算
    for elem in class_lables:
        class_matrix = np.mat(data[np.argwhere()])

    # print(lables)
    # lab = set()
    # for elem in lables:
    #     lab.add(elem)
    # print(lab)
    # print(len(title))
    # print(title)

    return np.array(title), features, lables


# 遗传算法
#
# 输入参数：
#   - features_name_t 特征名，顺序和特征值一致
#   - features_data_t 特征数据集，顺序和特征名一致
#   - lables_data_t 标签集，顺序和特征数据集一致
#
# 返回值：
#   - best_people_t 适应度最大的个体
#   - best_fitness_t 适应度的最大值
#   - fitness_change_t 每次迭代，最大适应度值的变化
#   - population_t 迭代完成后的种群
def GA(features_name_t, features_data_t, lables_data_t):
    # 特征总数量
    nr_features = len(features_name_t)

    # 根据输入的参数，随机生成一个初始种群
    population_t = np.zeros((init_pop, nr_features), dtype=np.int)  # 初始化种群，初始种群数量为200
    for i in range(init_pop):  # 定义种群的个体数为 n
        # a = np.zeros(nr_features - nr_select_d)  # 生成未被选择的特征
        # b = np.ones(nr_select_d)  # 将选择的d维特征定义为个体c中的1
        c = np.append(np.zeros(nr_features - nr_select, dtype=np.int), np.ones(nr_select, dtype=np.int))
        c = (np.random.permutation(c.T)).T  # 随机生成一个d维的个体
        population_t[i] = c  # 初代的种群为 population，共有n个个体

    # 遗传算法的迭代次数为ites
    fitness_change_t = np.zeros(ites, dtype=np.int)
    fitness = np.zeros(init_pop, dtype=np.int)
    # for i in range(init_pop):
    #     fitness = np.zeros(ites)  # fitness为每一个个体的适应度值
    #     for j in range(init_pop):
    #         fitness[j] = Jd(population_t[j])  # 计算每一个体的适应度值
    #     population_t = selection(population_t, fitness)  # 通过概率选择产生新一代的种群
    #     population_t = crossover(population_t)  # 通过交叉产生新的个体
    #     population_t = mutation(population_t)  # 通过变异产生新个体
    #     fitness_change_t[i] = max(fitness)  # 找出每一代的适应度最大的染色体的适应度值

    # 随着迭代的进行，每个个体的适应度值应该会不断增加，所以总的适应度值fitness求平均应该会变大

    best_fitness_t = max(fitness)
    best_people_t = population_t[fitness.argmax()]
    # print(type(best_people_t))
    fields_name_t = features_name_t[np.argwhere(best_people_t == 1)]

    return best_people_t, best_fitness_t, fitness_change_t, population_t, fields_name_t


# # 个体适应度函数 Jd(x)，x是d维特征向量(1*4维的行向量,1表示选择该特征)
# def Jd(x):
#     # 从特征向量x中提取出相应的特征
#     Feature = np.zeros(d)  # 数组Feature用来存 x选择的是哪d个特征
#     k = 0
#     for i in range(4):
#         if x[i] == 1:
#             Feature[k] = i
#             k += 1
#
#     # 将4个特征从原数据集中取出对应索引的特征数据列重组成一个150*d的矩阵iris3
#     iris3 = np.zeros((150, 1))
#     for i in range(d):
#         p = Feature[i]
#         p = p.astype(int)
#         q = iris2[:, p]
#         q = q.reshape(150, 1)
#         iris3 = np.append(iris3, q, axis=1)
#     iris3 = np.delete(iris3, 0, axis=1)
#
#     # 求类间离散度矩阵Sb
#     iris3_1 = iris3[0:50, :]  # iris数据集分为三类
#     iris3_2 = iris3[50:100, :]
#     iris3_3 = iris3[100:150, :]
#     m = np.mean(iris3, axis=0)  # 总体均值向量
#     m1 = np.mean(iris3_1, axis=0)  # 第一类的均值向量
#     m2 = np.mean(iris3_2, axis=0)  # 第二类的均值向量
#     m3 = np.mean(iris3_3, axis=0)  # 第二类的均值向量
#     m = m.reshape(d, 1)  # 将均值向量转换为列向量以便于计算
#     m1 = m1.reshape(d, 1)
#     m2 = m2.reshape(d, 1)
#     m3 = m3.reshape(d, 1)
#     Sb = ((m1 - m).dot((m1 - m).T) + (m2 - m).dot((m2 - m).T) + (m3 - m).dot((m3 - m).T)) / 3  # 除以类别个数
#
#     # 求类内离散度矩阵Sw
#     S1 = np.zeros((d, d))
#     S2 = np.zeros((d, d))
#     S3 = np.zeros((d, d))
#     for i in range(50):
#         S1 += (iris3_1[i].reshape(d, 1) - m1).dot((iris3_1[i].reshape(d, 1) - m1).T)
#     S1 = S1 / 50
#     for i in range(50):
#         S2 += (iris3_2[i].reshape(d, 1) - m2).dot((iris3_2[i].reshape(d, 1) - m2).T)
#     S2 = S2 / 50
#     for i in range(50):
#         S3 += (iris3_3[i].reshape(d, 1) - m3).dot((iris3_3[i].reshape(d, 1) - m3).T)
#     S3 = S3 / 50
#
#     Sw = (S1 + S2 + S3) / 3
#
#     # 计算个体适应度函数 Jd(x)
#     J1 = np.trace(Sb)
#     J2 = np.trace(Sw)
#     Jd = J1 / J2
#
#     return Jd


if __name__ == '__main__':
    features_name, features_data, lables_data = read_data()
    print("读入样本{}个，样本特征数量为{}.\n".format(len(lables_data), len(features_name)))
    print("算法运行中...")
    best_example, best_fitness, fitness_change, best_population, fields_name = GA(features_name, features_data,
                                                                                  lables_data)
    # choice = np.zeros(nr_select)
    # k = 0
    # print("在取%d维的时候，通过遗传算法得出的最优适应度值为：%.6f" % (nr_select, best_fitness))
    # print("选出的最优染色体为：")
    # print(best_example)
    # for i in range(nr_fields):
    #     if best_example[i] == 1:
    #         choice[k] = i + 1
    #         k += 1
    # print("选出的最优特征为：")
    # print(choice)

    # test
    # feature_name, features = read_data(input_path)
    # print(feature_name)
    # # feature_name.pop(2)
    # for elem in reject_list:
    #     feature_name.remove(elem)
    # print(feature_name)
    # print(type(feature_name))
    # print(type(features))
    # print(features[:10, :])
    # print('len(feature_name):{}'.format(len(feature_name)))
    # print('len(features):{}'.format(len(features)))
    # print(features[:10, [-2]])
    # head = features[:5, :]
    # # print(head)
    # head = np.delete(head, 2, 1)
    # print(head)

    # default_feature = set()
    # features = np.delete(features, -1, 1)
    # features = np.delete(features, 2, 1)
    # for row in features:
    #     # print(row)
    #     for index in range(len(row)):
    #         try:
    #             number = float(row[index])
    #         except:
    #             print(row)
    #             break
    #         if number < 0:
    #             default_feature.add(index)
    #             # print(row)
    #             # break
    # print(default_feature)
    read_data(input_path)
