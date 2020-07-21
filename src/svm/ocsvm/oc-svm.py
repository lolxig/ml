# 尝试自己写一个ocsvm
# 框架参考smo算法

import random
from numpy import *


# 从数据文件中载入数据并进行切分
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# 随机抽两个参数，保持两个参数不重复
# 这里简化了参数的抽取过程，直接遍历了整个数据集
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# 对输入的alpha_j进行范围检查，保证输出的alpha_j在[L,H]的范围内
def clipAlpha(aj, L, H):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


# 参数说明
#   - dataMatIn：输入特征集
#   - classLabels：输入标签集
#   - v：惩罚参数，范围为(0,1]
#   - toler：容错率
#   - maxIter：最大迭代次数
def simpleOcsvm(dataMatIn, classLabels, v, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMatrix = mat(classLabels).transpose()  # 转置
    m, n = shape(dataMatrix)  # m为输入数据的个数，n为输入向量的维数
    vl = 1.0 / v * m  # 正则化参数
    p = 0  # 偏移项
    alphas = mat(zeros((m, 1)))  # 初始化参数，确定m个alpha
    iter = 0  # 用于计算迭代次数

    while iter < maxIter:  # 当迭代次数小于最大迭代次数时（外循环）
        alphaPairsChanged = 0  # 初始化alpha的改变量为0
        for i in range(m):  # 外循环
            fXi = float(multiply(alpha, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 计算f(xi)
            Ei = fXi - float(labelMat[i])  # 计算f(xi)与标签之间的误差




            # 如果不满足KKT条件，则优化
            if ((Oi - p) * alphas[i] > 0) or ((p - Oi)*(vl - alphas[i]) > 0):




            if ((labelMat[i] * Ei < -toler) and (alpha[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alpha[i] > 0)):
                j = selectJrand(i, m)  # 随机选择一个j与i配对
                fXj = float(multiply(alpha, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b  # 计算f(xj)
                Ej = fXj - float(labelMat[j])  # 计算j的误差
                alphaIold = alpha[i].copy()  # 保存原来的alpha(i)
                alphaJold = alpha[j].copy()
                if labelMat[i] != labelMat[j]:  # 保证alpha在0到c之间
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    print('L=H')
                    continue
                eta = 2 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta=0')
                    continue
                alpha[j] -= labelMat[j] * (Ei - Ej) / eta
                alpha[j] = clipAlpha(alpha[j], H, L)  # 调整大于H或小于L的alpha
                if abs(alpha[j] - alphaJold) < 0.0001:
                    print('j not move enough')
                    continue
                alpha[i] += labelMat[j] * labelMat[i] * (alphaJold - alpha[j])
                b1 = b - Ei - labelMat[i] * (alpha[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alpha[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T  # 设置b
                b2 = b - Ej - labelMat[i] * (alpha[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alpha[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alpha[i]) and (C > alpha[j]):
                    b = b1
                elif (0 < alpha[j]) and (C > alpha[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphaPairsChanged += 1
                print('iter:%d i:%d,pairs changed%d' % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print('iteraction number:%d' % iter)
    return alpha, p