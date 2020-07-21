# 目标求解 2*sin(x)+cos(x)的最大值

import random
import math
import matplotlib.pyplot as plt


# 初始化生成chromosome_length大小的population_size个个体的二进制基因型种群
# population_size: 种群大小
# chromosome_length: 每个个体的基因数目
def species_origin(population_size, chromosome_length):
    # 二维列表，包含染色体和基因
    population = [[]]
    # 对种群遍历
    for i in range(population_size):
        # 染色体暂存器
        temporary = []
        # 对个体遍历
        for j in range(chromosome_length):
            # 随机产生一个染色体，由二进制数组成
            temporary.append(random.randint(0, 1))
        # 将个体添加到种群中
        population.append(temporary)
    # 返回生成的种群
    return population


# 编码，从二进制到十进制
# input: 种群，个体基因数量
def translation(population, chromosome_length):
    # 转换后的种群缓存
    temporary = []
    for i in range(len(population)):
        total = 0
        for j in range(chromosome_length):
            # 从第一个基因开始，每位对2求幂，再求和
            # 如：0101转换成5
            total += population[i][j] * math.pow(2, j)
        temporary.append(total)
    # 返回编码后的种群
    return temporary


# 目标函数相当于环境对个体进行筛选，这里是2*sin(x)+cos(x)
def function(population, chromosome_length, max_value):
    function1 = []
    # 暂存种群中的所有个体(十进制)
    temporary = translation(population, chromosome_length)
    for i in range(len(temporary)):
        # 一个基因代表一个决策变量，其算法是先转化成十进制，然后再除以2的基因个数此房减1(固定值)
        # 这里x的生成方式，只是一种编码规则，对最后的优化结果没有影响，但是对优化速度和平滑性有影响，选的好的话，可以加快优化效果
        x = temporary[i] * max_value / (math.pow(2, chromosome_length) - 1)
        # 这里将2*sin(x)+cos(x)作为目标函数，也就是适应函数
        function1.append(2 * math.sin(x) + math.cos(x))
    return function1


# 只保留非负值得适应度/函数值
def fitness(function1):
    # 保留通用函数写法
    fitness1 = []
    min_fitness = mf = 0
    for i in range(len(function1)):
        if fitness1[i] + mf > 0:
            temporary = mf + function1[i]
        else:
            temporary = 0.0
        fitness1.append(temporary)
    return fitness1


# 计算适应度之和
def sum(fitness1):
    total = 0
    for i in range(len(fitness1)):
        total += fitness1[i]
    return total


# 计算适应度斐波那契列表，这里是为了求出累积的适应度
def cumsum(fitness1):
    # range(start, stop, [step])
    # step为-1时倒计数
    # 从len(fitness1) - 2倒数到0(包含0)
    for i in range(len(fitness1) - 2, -1, -1):
        total = 0
        j = 0
        while j <= i:
            total += fitness1[j]
            j += 1
        fitness1[i] = total
        fitness1[len(fitness1) - 1] = 1


# 用轮盘赌选择种群中个体适应度最大的个体
def selection(population, fitness1):
    # 单个公式暂存器
    new_fitness = []
    # 将所有的适用度求和
    total_fitness = sum(fitness1)
    # 将所有个体的适应度概率化，类似于softmax
    for i in range(len(fitness1)):
        new_fitness.append(fitness1[i] / total_fitness)
    # 将所有个体的适应度划分成区间
    cumsum(new_fitness)
    # 存活的种群
    ms = []
    # 求出种群的长度
    # 根据随机数确定哪几个能存活
    population_length = pop_len = len(population)

    # 产生种群个数的随机值
    for i in range(pop_len):
        ms.append(random.random())
    # 存活的种群排序
    ms.sort()

    # 轮盘赌
    fitin = 0
    newin = 0
    new_population = new_pop = population
    while newin < pop_len:
        if ms[newin] < new_fitness[fitin]:
            new_pop[newin] = population[fitin]
            newin += 1
        else:
            fitin += 1
    population = new_pop


# 交叉
# pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
def crossover(population, pc):
    pop_len = len(population)

    for i in range(pop_len - 1):
        # 在种群内随机生成单点交叉点
        # randint(a,b): 返回一个[a, b]区间内的随机值
        cpoint = random.randint(0, len(population[0]))

        temporary1 = []
        temporary2 = []

        # 将temporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因
        # 然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
        # extend()：在列表末尾追加另一个序列中的多个值
        temporary1.extend(population[i][0:cpoint])
        temporary1.extend(population[i + 1][cpoint:len(population[i])])

        # 将temporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因
        # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
        temporary2.extend(population[i + 1][0:cpoint])
        temporary2.extend(population[i][cpoint:len(population[i])])

        # 第i个染色体和第i+1个染色体基因重组/交叉完成
        population[i] = temporary1
        population[i + 1] = temporary2


# 变异
# pm是概率阈值
def mutation(population, pm):
    # 求出种群中的个体的个数
    px = len(population)
    # 个体中基因的个数
    py = len(population[0])

    for i in range(px):
        # 如果小于阈值就变异
        if random.random() < pm:
            # 生成0到py-1的随机数
            mpoint = random.randint(0, py - 1)
            # 将mpoint个基因进行单点随机变异，变为0或者1
            if population[i][mpoint] == 1:
                population[i][mpoint] = 0
            else:
                population[i][mpoint] = 1


# 其他
