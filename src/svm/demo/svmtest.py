# svm算法的实现
from numpy import *
import random
from time import *


# 输出dataArr(m*n),labelArr(1*m)其中m为数据集的个数
# dataMat为数据集特征矩阵，labelMat为标签向量
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')  # 去除制表符，将数据分开
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 数组矩阵
        labelMat.append(float(lineArr[2]))  # 标签
    return dataMat, labelMat


# 随机找一个和i不同的j
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# 调整大于H或小于L的alpha的值，使alpha的值落到[L,H]的区间内
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()  # 转置
    b = 0
    m, n = shape(dataMatrix)  # m为输入数据的个数，n为输入向量的维数
    alpha = mat(zeros((m, 1)))  # 初始化参数，确定m个alpha
    iter = 0  # 用于计算迭代次数
    while iter < maxIter:  # 当迭代次数小于最大迭代次数时（外循环）
        alphaPairsChanged = 0  # 初始化alpha的改变量为0
        for i in range(m):  # 外循环
            fXi = float(multiply(alpha, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 计算f(xi)
            Ei = fXi - float(labelMat[i])  # 计算f(xi)与标签之间的误差
            if ((labelMat[i] * Ei < -toler) and (alpha[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alpha[i] > 0)):  # 如果可以进行优化
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
    return b, alpha


# 定义径向基函数
# X: mxn的矩阵
# A: 1xn的行向量
def kernelTrans(X, A, kTup):  # 定义核转换函数（径向基函数）
    m, n = shape(X)
    # K: mx1的列向量
    K = mat(zeros((m, 1)))
    # 线性核，得到Gram矩阵
    if kTup[0] == 'lin':
        K = X * A.T  # 线性核K为m*1的矩阵
    # 得到高斯核函数的Gram矩阵
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    # 出错
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


# 整个数据的结构体
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        # 输入数据集的特征集
        self.X = dataMatIn
        # 输入数据集的标签集
        self.labelMat = classLabels
        # 正则约束C
        self.C = C
        # 容错率
        self.tol = toler
        # 数据集的数据条数
        self.m = shape(dataMatIn)[0]
        # 初始的alpha向量
        self.alphas = mat(zeros((self.m, 1)))
        # 初始的偏移项b
        self.b = 0
        # 差值缓存矩阵，第一列为有效标志位
        self.eCache = mat(zeros((self.m, 2)))
        # Gram缓存矩阵
        self.K = mat(zeros((self.m, self.m)))
        # 生成初始的Gram矩阵
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


# 计算特定的差值
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 用启发式选择第二个参数
def selectJ(i, oS, Ei):
    # 保留差值最大的索引
    maxK = -1
    # 最大差值的差值的绝对值
    maxDeltaE = 0
    # 最大的差值
    Ej = 0
    # 更新差值缓存
    oS.eCache[i] = [1, Ei]
    # 扫描差值缓存列表的有效位
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    # 若缓存列表不为空
    if (len(validEcacheList)) > 1:
        # 对差值缓存列表进行扫描
        for k in validEcacheList:
            # 跳过等于i的
            if k == i:
                continue
            # 计算扫描到的差值
            Ek = calcEk(oS, k)
            # 得到差值的差值
            deltaE = abs(Ei - Ek)
            # 选择最大的差值
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    # 若缓存列表为空，则随机选一个
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 对第k个差值缓存进行更新
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 内循环函数
def innerL(i, oS):
    # 对第i个alpha计算差值
    Ei = calcEk(oS, i)
    # 如果不满足KKT条件，则对其进行更新
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 用启发式选择第二个变量
        j, Ej = selectJ(i, oS, Ei)
        # 存储旧的alpha值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 计算边界上下限
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        # 若边界上下限相等，则无法进行优化，返回
        if L == H:
            print("L==H")
            return False
        # 计算-(K11+K22-2K12)，即分母，而-eta必须为正数
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        # eta等于0计算会很麻烦，按照错误值忽略
        if eta >= 0:
            print("eta>=0")
            return False
        # 更新alpha_2
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 将更新后的值进行范围限定
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 将更新后的值，进行差值更新
        updateEk(oS, j)
        # 若更新的幅度过小，则跳过alpha_1的更新
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return False
        # 通过新的alpha_j更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 将更新后的alpha_i更新到差值列表中去
        updateEk(oS, i)
        # 计算偏移项
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        # 更新成功
        return True
    else:
        return False


# smoP函数用于计算超平的alpha,b
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # 完整的Platter SMO
    # 将数据存储到一个数据结构里面
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    # 计算循环的次数
    iter = 0
    # 控制是否遍历整个数据集
    entireSet = True
    # 用来探查alpha的是否在该轮中进行了优化
    alphaPairsChanged = False
    # 外循环
    while (iter < maxIter) and (alphaPairsChanged or entireSet):
        alphaPairsChanged = False
        # 遍历整个数据集对alpha和b进行查找和优化
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged = innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %s" % (iter, i, str(alphaPairsChanged)))
            iter += 1
        # 对非边界上的数据进行扫描
        else:
            # mat.A将矩阵转换为数组类型(numpy 的narray)
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 遍历不在边界上的数据
            for i in nonBoundIs:
                alphaPairsChanged = innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %s" % (iter, i, alphaPairsChanged))
            iter += 1
        # 若遍历过了整个数据集，下一次不再遍历
        if entireSet:
            entireSet = False
        # 若没有遍历整个数据集且参数在本轮中没有进行优化，则下次遍历整个数据集
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


# calcWs用于计算权重值w
def calcWs(alphas, dataArr, classLabels):  # 计算权重W
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


# 值得注意的是测试准确与k1和C的取值有关。
def testRbf(k1=1.3):  # 给定输入参数K1
    # 测试训练集上的准确率
    dataArr, labelArr = loadDataSet('testSetRBF.txt')  # 导入数据作为训练集
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]  # 找出alphas中大于0的元素的位置
    # 此处需要说明一下alphas.A的含义
    sVs = datMat[svInd]  # 获取支持向量的矩阵，因为只要alpha中不等于0的元素都是支持向量
    labelSV = labelMat[svInd]  # 支持向量的标签
    print("there are %d Support Vectors" % shape(sVs)[0])  # 输出有多少个支持向量
    # 数据组的矩阵形状表示为有m个数据，数据维数为n
    m, n = shape(datMat)
    # 计算错误的个数
    errorCount = 0
    # 开始分类，是函数的核心
    for i in range(m):
        # 对该行数据进行高斯核转换
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        # 计算预测结果y的值
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        # 利用符号判断类别
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
        # sign（a）为符号函数：若a>0则输出1，若a<0则输出-1.
    print("the training error rate is: %f" % (float(errorCount) / m))
    # 2、测试测试集上的准确率
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    # labelMat = mat(labelArr).transpose()此处可以不用
    datMat = mat(dataArr)
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


def test():
    dataArr, labelArr = loadDataSet('testSet.txt')
    # print(labelArr)
    # labelArr
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print("\nover.")
    print(b)
    print(alphas)


def main():
    t1 = time()
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.01, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    testRbf()
    t2 = time()
    print("程序所用时间为%ss" % (t2 - t1))


if __name__ == '__main__':
    main()
    # test()
