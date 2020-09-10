import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

input_path = 'E:\\论文\\小论文两篇\\GA实验\\train_data_scaler.csv'

# 读取数据文件并进行分离
# ids_data = pd.read_csv(input_path, encoding='utf-8').drop(columns='Timestamp')
# ids_data_Benign = ids_data[ids_data.loc[:, 'Label'] == 'Benign'].drop(columns='Label')
# ids_data_Infilteration = ids_data[ids_data.loc[:, 'Label'] == 'Infilteration'].drop(columns='Label')

ids_data = pd.read_csv(input_path, sep='\t', encoding='utf-8')

ids_data_1 = ids_data[ids_data.loc[:, 'Label'] == 'one'].drop(columns='Label')
ids_data_2 = ids_data[ids_data.loc[:, 'Label'] == 'two'].drop(columns='Label')
ids_data_3 = ids_data[ids_data.loc[:, 'Label'] == 'three'].drop(columns='Label')
ids_data_4 = ids_data[ids_data.loc[:, 'Label'] == 'four'].drop(columns='Label')
ids_data_5 = ids_data[ids_data.loc[:, 'Label'] == 'five'].drop(columns='Label')

# 取部分数据
# len_Benign = int(ids_data_Benign.__len__() / 100)
# len_Infilteration = int(ids_data_Infilteration.__len__() / 100)
# len_total = len_Benign + len_Infilteration

# r_Benign = random.sample(range(0, ids_data_Benign.__len__()), len_Benign)
# r_Infilteration = random.sample(range(0, ids_data_Infilteration.__len__()), len_Infilteration)

# ids_data_Benign = ids_data_Benign.iloc[r_Benign, :]
# ids_data_Infilteration = ids_data_Infilteration.iloc[r_Infilteration, :]
# ids_data = pd.cancat(ids_data_Benign, ids_data_Infilteration)

# ids_data = ids_data.iloc[np.append(ids_data_Benign._stat_axis.values, ids_data_Infilteration._stat_axis.values), :]

rat = 100

len_1 = int(ids_data_1.__len__() / rat)
len_2 = int(ids_data_2.__len__() / rat)
len_3 = int(ids_data_3.__len__() / rat)
len_4 = int(ids_data_4.__len__() / rat)
len_5 = int(ids_data_5.__len__() / rat)

len_total = len_1 + len_2 + len_3 + len_4 + len_5

r_len_1 = random.sample(range(0, ids_data_1.__len__()), len_1)
r_len_2 = random.sample(range(0, ids_data_2.__len__()), len_2)
r_len_3 = random.sample(range(0, ids_data_3.__len__()), len_3)
r_len_4 = random.sample(range(0, ids_data_4.__len__()), len_4)
r_len_5 = random.sample(range(0, ids_data_5.__len__()), len_5)

ids_data_1 = ids_data_1.iloc[r_len_1, :]
ids_data_2 = ids_data_2.iloc[r_len_2, :]
ids_data_3 = ids_data_3.iloc[r_len_3, :]
ids_data_4 = ids_data_4.iloc[r_len_4, :]
ids_data_5 = ids_data_5.iloc[r_len_5, :]

l = np.append(ids_data_1._stat_axis.values, ids_data_2._stat_axis)
l = np.append(l, ids_data_3._stat_axis)
l = np.append(l, ids_data_4._stat_axis)
l = np.append(l, ids_data_5._stat_axis)

ids_data = ids_data.iloc[l, :]

# 取得各个样本的数量和总体样本的数量
# len_total = ids_data.__len__()
# len_Benign = ids_data_Benign.__len__()
# len_Infilteration = ids_data_Infilteration.__len__()

# 求得各个类在总体中的频率，代替概率
# pr_Benign = len_Benign / len_total
# pr_Infilteration = len_Infilteration / len_total

pr_1 = len_1 / len_total
pr_2 = len_2 / len_total
pr_3 = len_3 / len_total
pr_4 = len_4 / len_total
pr_5 = len_5 / len_total

# 取得各个样本和总体样本的均值向量
# mean_total = ids_data.mean()
# mean_Benign = ids_data_Benign.mean()
# mean_Infilteration = ids_data_Infilteration.mean()

# 样本特征集字段数目
nr_feature = ids_data_1.shape[1]

# 将各类数据转换为矩阵，方便后续的求解
# ids_data = np.mat(ids_data)
# ids_data_Benign = np.mat(ids_data_Benign)
# ids_data_Infilteration = np.mat(ids_data_Infilteration)

# 交叉概率、变异概率、迭代次数、初始种群大小
pc = 0.25
pm = 0.01
t = 50
n = 100

# 待选择的特征数
nr_select = 2


# 遗传算法
# d: 待选择的字段数目
def GA(d):
    if d > nr_feature:
        print('所选择的特征数量大于数据集特征的总数量，请检查')
        exit(1)

    # 随机生成一个种群，每个个体的特征都是随机选择的
    population_t = np.zeros((n, nr_feature), dtype=np.int)
    for i in range(n):  # 定义种群的个体数为 n
        a = np.zeros(nr_feature - d, dtype=np.int)  # 生成未被选择的特征
        b = np.ones(d, dtype=np.int)  # 将选择的d维特征定义为个体c中的1
        c = np.append(a, b)
        c = (np.random.permutation(c.T)).T  # 随机生成一个d维的个体
        population_t[i] = c  # 初代的种群为 population，共有n个个体

    # 遗传算法的迭代次数为t
    fitness_change_t = np.zeros(t)
    fitness = np.zeros(n)
    for i in range(t):
        fitness = np.zeros(n)  # fitness为每一个个体的适应度值
        for j in range(n):
            fitness[j] = Jd(population_t[j], d)  # 计算每一个体的适应度值
        population_t = selection(population_t, fitness)  # 通过轮盘赌选择产生新一代的种群
        population_t = crossover(population_t)  # 通过交叉产生新的个体
        population_t = mutation(population_t)  # 通过变异产生新个体
        fitness_change_t[i] = max(fitness)  # 找出每一代的适应度最大的染色体的适应度值

    # 随着迭代的进行，每个个体的适应度值应该会不断增加，所以总的适应度值fitness求平均应该会变大

    best_fitness_t = max(fitness)
    best_people_t = population_t[fitness.argmax()]

    # 返回：
    # - best_people_t 最大适度的染色体
    # - best_fitness_t 最大适度值
    # - fitness_change_t 适度值在训练过程中的变化
    # - population_t 最大适度值的种群
    return best_people_t, best_fitness_t, fitness_change_t, population_t


# 个体适应度函数 Jd(x)
# - x: 个体
# - d: 待选择的特征数量
def Jd(x, d):
    feature_index = np.zeros(d, dtype=np.int)  # 数组Feature用来存 x选择的是哪d个特征
    # 从特征向量x中提取出被选中的特征索引
    k = 0
    for i in range(nr_feature):
        if x[i] == 1:
            feature_index[k] = i
            k += 1

    # 获取所选择的特征对应的列数据
    # ids_data_select = ids_data.iloc[:, feature_index]
    # ids_data_Benign_select = ids_data_Benign.iloc[:, feature_index]
    # ids_data_Infilteration_select = ids_data_Infilteration.iloc[:, feature_index]

    ids_data_select = ids_data.iloc[:, feature_index]
    ids_data_1_select = ids_data_1.iloc[:, feature_index]
    ids_data_2_select = ids_data_2.iloc[:, feature_index]
    ids_data_3_select = ids_data_3.iloc[:, feature_index]
    ids_data_4_select = ids_data_4.iloc[:, feature_index]
    ids_data_5_select = ids_data_5.iloc[:, feature_index]

    # 求得获取的列数据的均值向量
    # mean_total = np.mat(ids_data_select.mean()).reshape(d, 1)
    # mean_Benign = np.mat(ids_data_Benign_select.mean()).reshape(d, 1)
    # mean_Infilteration = np.mat(ids_data_Infilteration_select.mean()).reshape(d, 1)

    mean_total = np.mat(ids_data_select.mean()).reshape(d, 1)
    mean_1 = np.mat(ids_data_1_select.mean()).reshape(d, 1)
    mean_2 = np.mat(ids_data_2_select.mean()).reshape(d, 1)
    mean_3 = np.mat(ids_data_3_select.mean()).reshape(d, 1)
    mean_4 = np.mat(ids_data_4_select.mean()).reshape(d, 1)
    mean_5 = np.mat(ids_data_5_select.mean()).reshape(d, 1)

    # 求类间离散度矩阵Sb
    # Sb = (mean_Benign - mean_total).dot((mean_Benign - mean_total).T) * pr_Benign \
    #      + (mean_Infilteration - mean_total).dot((mean_Infilteration - mean_total).T) * pr_Infilteration

    Sb = (mean_1 - mean_total).dot((mean_1 - mean_total).T) * pr_1 \
         + (mean_2 - mean_total).dot((mean_2 - mean_total).T) * pr_2 \
         + (mean_3 - mean_total).dot((mean_3 - mean_total).T) * pr_3 \
         + (mean_4 - mean_total).dot((mean_4 - mean_total).T) * pr_4 \
         + (mean_5 - mean_total).dot((mean_5 - mean_total).T) * pr_5

    # 将选择的列数据转换成矩阵方便计算
    # ids_data_Benign_select = np.mat(ids_data_Benign_select)
    # ids_data_Infilteration_select = np.mat(ids_data_Infilteration_select)

    ids_data_1_select = np.mat(ids_data_1_select)
    ids_data_2_select = np.mat(ids_data_2_select)
    ids_data_3_select = np.mat(ids_data_3_select)
    ids_data_4_select = np.mat(ids_data_4_select)
    ids_data_5_select = np.mat(ids_data_5_select)

    # 求类内离散度矩阵Sw
    # S_Benign = np.zeros((d, d))
    # S_Infilteration = np.zeros((d, d))

    S_1 = np.zeros((d, d))
    S_2 = np.zeros((d, d))
    S_3 = np.zeros((d, d))
    S_4 = np.zeros((d, d))
    S_5 = np.zeros((d, d))

    for elem in ids_data_1_select:
        S_1 += (elem.reshape(d, 1) - mean_1).dot((elem.reshape(d, 1) - mean_1).T)
    S_1 = S_1 / len_1 * pr_1

    for elem in ids_data_2_select:
        S_2 += (elem.reshape(d, 1) - mean_2).dot((elem.reshape(d, 1) - mean_2).T)
    S_2 = S_2 / len_2 * pr_2

    for elem in ids_data_3_select:
        S_3 += (elem.reshape(d, 1) - mean_3).dot((elem.reshape(d, 1) - mean_3).T)
    S_3 = S_3 / len_3 * pr_3

    for elem in ids_data_4_select:
        S_4 += (elem.reshape(d, 1) - mean_4).dot((elem.reshape(d, 1) - mean_4).T)
    S_4 = S_4 / len_4 * pr_4

    for elem in ids_data_5_select:
        S_5 += (elem.reshape(d, 1) - mean_5).dot((elem.reshape(d, 1) - mean_5).T)
    S_5 = S_5 / len_5 * pr_5

    # for elem in ids_data_Benign_select:
    #     S_Benign += (elem.reshape(d, 1) - mean_Benign).dot((elem.reshape(d, 1) - mean_Benign).T)
    # S_Benign = S_Benign / len_Benign * pr_Benign
    #
    # for elem in ids_data_Infilteration_select:
    #     S_Infilteration += (elem.reshape(d, 1) - mean_Infilteration).dot((elem.reshape(d, 1) - mean_Infilteration).T)
    # S_Infilteration = S_Infilteration / len_Infilteration * pr_Infilteration

    # Sw = S_Benign + S_Infilteration

    Sw = S_1 + S_2 + S_3 + S_4 + S_5

    # 计算个体适应度函数 Jd(x)
    # j1 = np.trace(Sb)
    # j2 = np.trace(Sw)
    # Jd = j1 / j2
    return np.trace(Sb) / np.trace(Sw)


# 轮盘赌选择
def selection(population, fitness):
    fitness_sum = np.zeros(n)
    for i in range(n):
        if i == 0:
            fitness_sum[i] = fitness[i]
        else:
            fitness_sum[i] = fitness[i] + fitness_sum[i - 1]
    for i in range(n):
        fitness_sum[i] = fitness_sum[i] / sum(fitness)

    # 选择新的种群
    population_new = np.zeros((n, nr_feature), dtype=np.int)
    for i in range(n):
        rand = np.random.uniform(0, 1)
        for j in range(n):  # 可以使用二分法改进(二分法改进较麻烦，后续改进)
            if j == 0:
                if rand <= fitness_sum[j]:
                    population_new[i] = population[j]
            else:
                if fitness_sum[j - 1] < rand <= fitness_sum[j]:
                    population_new[i] = population[j]
    return population_new


# 交叉操作
def crossover(population):
    father = population[0:int(n * pc), :]  # 对pc比率的个体进行交叉
    mother = population[len(father):, :]
    np.random.shuffle(father)  # 将父代个体按行打乱以随机配对
    np.random.shuffle(mother)
    for i in range(len(father)):
        father_1 = father[i]
        mother_1 = mother[i]
        one_zero = []
        zero_one = []
        for j in range(nr_feature):  # 遍历每一个特征
            if father_1[j] == 1 and mother_1[j] == 0:
                one_zero.append(j)
            if father_1[j] == 0 and mother_1[j] == 1:
                zero_one.append(j)
        length1 = len(one_zero)
        length2 = len(zero_one)
        length = max(length1, length2)
        half_length = int(length / 2)  # half_length为交叉的位数
        for k in range(half_length):  # 进行交叉操作
            p = one_zero[k]
            q = zero_one[k]
            father_1[p] = 0
            mother_1[p] = 1
            father_1[q] = 1
            mother_1[q] = 0
        father[i] = father_1  # 将交叉后的个体替换原来的个体
        mother[i] = mother_1
    population = np.append(father, mother, axis=0)
    return population


# 变异操作
def mutation(population):
    for i in range(n):
        c = np.random.uniform(0, 1)
        if c <= pm:
            mutation_s = population[i]
            zero = []  # zero存的是变异个体中第几个数为0
            one = []  # one存的是变异个体中第几个数为1
            for j in range(60):
                if mutation_s[j] == 0:
                    zero.append(j)
                else:
                    one.append(j)

            if (len(zero) != 0) and (len(one) != 0):
                a = np.random.randint(0, len(zero))  # e是随机选择由0变为1的位置
                b = np.random.randint(0, len(one))  # f是随机选择由1变为0的位置
                e = zero[a]
                f = one[b]
                mutation_s[e] = 1
                mutation_s[f] = 0
                population[i] = mutation_s

    return population


if __name__ == '__main__':

    # for count in range(10):
    best_people, best_fitness, fitness_change, best_population = GA(nr_select)
    choice = np.zeros(nr_select)
    k = 0
    print("在取%d维的时候，通过遗传算法得出的最优适应度值为：%.6f" % (nr_select, best_fitness))
    print("选出的最优染色体为：")
    print(best_people)
    for j in range(nr_feature):
        if best_people[j] == 1:
            choice[k] = j + 1
            k += 1
    print("选出的最优特征为：")
    print(choice)
    print('------------------------------------------------------')

    # print('十轮运行完毕.')

    # # 画图
    x = np.arange(0, t, 1)
    plt.xlabel('dimension')
    plt.ylabel('fitness')
    plt.ylim((min(fitness_change), max(fitness_change)))  # y坐标的范围
    plt.plot(x, fitness_change, 'b')
    plt.show()
