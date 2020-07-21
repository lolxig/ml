import random
from numpy import *

data_path = 'testSet.txt'


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


# 小测试
def test_1():
    dataArr, labelArr = loadDataSet(data_path)
    # print(labelArr)
    # labelArr
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    print("\nover.")
    print(b)
    print(alphas[alphas > 0])
    print('\nsupport vector:')
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataArr[i], labelArr[i])
    w = calcWs(alphas, dataArr, labelArr)
    print('\nws:')
    print(w)


# 简化SMO算法
# 参数说明：
#   - dataMatIn     数据集
#   - classLabel    类别标签
#   - C             正则项C
#   - toler         容错率
#   - maxIter       最大迭代次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)  # 将输入数据集转换为矩阵
    labelMatrix = mat(classLabels).transpose()  # 将标签转置为列向量
    b = 0
    m, n = shape(dataMatrix)  # m条数据，每条数据有n个特征
    alphas = mat(zeros((m, 1)))  # 初始化参数，确定m个alpha | zeros((m,n))：产生一个m行n列元素全为0的多维数组 | 这里产生一个m行的列向量alphas
    iter = 0  # 用于计算迭代次数，初始化为0
    while iter < maxIter:
        alphaPairsChanged = 0  # 初始化alpha的改变次数为0
        for i in range(m):
            # multiply(A,B)：让矩阵A和矩阵B对应的位置相乘并得到一个新矩阵，这里是让alpha_i乘以y_i得到一个行向量
            # multiply(alphas, labelMatrix).T为 (alpha_i y_i)^T，是1xm的行向量
            # dataMatrix * dataMatrix[i, :].T为 (x_i * x_j)内积，输出是一个mx1的列向量
            # fXi为模型在xi上的预测值
            fXi = float(multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 计算f(xi)
            # 预测值与真实值之间的差值Ei
            Ei = fXi - float(labelMatrix[i])
            # 满足条件即可优化参数
            if ((labelMatrix[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMatrix[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 随机选择一个j与i匹配
                fXj = float(multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b  # 计算f(xj)
                # 预测值与真实值之间的差值Ej
                Ej = fXj - float(labelMatrix[j])
                # 保存改变前的一对值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 在二值方框内调整L和H的值
                # 对应公式(7.104的上面)
                if labelMatrix[i] != labelMatrix[j]:  # 保证alpha在0到c之间
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L=H')
                    continue
                # 对应公司(7.107)，取负数
                eta = 2 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta=0')
                    continue
                # 对应公式(7.106)
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                # 对应公式(7.108)
                alphas[j] = clipAlpha(alphas[j], L, H)  # 调整大于H或小于L的alpha
                # 移动的梯度过小，跳过
                if abs(alphas[j] - alphaJold) < 0.0001:
                    print('j not move enough')
                    continue
                # 更新alpha_i的值
                # 对应公式(7.109)
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (alphaJold - alphas[j])
                # 更新b1的值
                # 公式(7.115)
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMatrix[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                # 公式(7.116)
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMatrix[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                # 根据条件选择b
                # 公式(7.116)-公式(7.117)之间的文字
                if (0 < alphas[i]) and (C > alphas[j]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                # alpha的改变次数加一
                alphaPairsChanged += 1
                print('iter:%d i:%d,pairs changed%d' % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print('iteraction number:%d' % iter)
    return b, alphas


# 建立一个对象来存储所有重要的值
class optStruct:
    def __init__(self, dataMatrixIn, classLabels, C, toler):
        self.X = dataMatrixIn
        self.labelMatrix = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatrixIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # 误差缓存 | 第一列是有效标志位，第二列是实际的E值


# 计算Ek
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMatrix).T * (oS.X * oS.X[k, :].T)) + oS.b
    fXk = float(multiply(oS.alphas, oS.labelMatrix).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMatrix[k])
    return Ek


# 选择第二个变量，标准是：希望alpha_2有足够大的变化
# 参数：
#   - i，第一个变量
#   - oS，数据集对象
#   - Ei，第一个变量对应的差值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 将第i个E标记为有效
    oS.eCache[i] = [1, Ei]
    # nonzero():获取非零元素的索引
    # validEcacheList:有效的缓存列索引
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    # 如果有差值缓存，则按照差值缓存表进行遍历
    if (len(validEcacheList)) > 1:
        # 遍历每一个有效的差值索引，找到最大的差值
        for k in validEcacheList:
            # 跳过Ei的计算
            if k == i:
                continue
            # 计算Ek
            Ek = calcEk(oS, k)
            # 求得Ei与Ek之间的距离
            deltaE = abs(Ei - Ek)
            # 找到最大的差值
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        # 遍历完毕，返回最大的差值以及第二个变量的索引
        return maxK, Ej
    # 如果没有差值缓存，随机获取第二个变量，然后计算Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = Ek


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMatrix[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or \
            (oS.labelMatrix[i] * Ei > oS.tol and oS.alphas[i] > oS.C):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMatrix[i] != oS.labelMatrix[j]:  # 保证alpha在0到c之间
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L=H')
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print('eta=0')
            return 0
        oS.alphas[j] -= oS.labelMatrix[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], L, H)
        if abs(oS.alphas[j] - alphaJold) < 0.0001:
            print('j not move enough')
            return 0
        oS.alphas[i] += oS.labelMatrix[j] * oS.labelMatrix[i] * (alphaJold - oS.alphas[j])
        b1 = oS.b - Ei - oS.labelMatrix[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMatrix[j] * (oS.alphas[j] - alphaJold) * \
             oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMatrix[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMatrix[j] * (oS.alphas[j] - alphaJold) * \
             oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[j]):
            b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            b = b2
        else:
            b = (b1 + b2) / 2
        return 1
    else:
        return 0


# 参数说明：
#   - dataMatIn     数据集
#   - classLabel    类别标签
#   - C             正则项C
#   - toler         容错率
#   - maxIter       最大迭代次数
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 循环退出条件：迭代次数达到最大值 | 遍历整个集合都没有对任何alpha进行修改
    while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            # 遍历所有的alpha
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # mat.A将矩阵转换为数组类型(numpy 的narray)
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 遍历所有不在边界上的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabel):
    X = mat(dataArr)
    labelMat = mat(classLabel).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


if __name__ == '__main__':
    test_1()
