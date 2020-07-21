import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from scipy.optimize import leastsq
from scipy import stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import CubicSpline
import math


if __name__ == "__main__":
    # a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
    # print(a)

    # L = [1, 2, 3, 4, 5, 6]
    # # print("L = ", L)
    # a = np.array(L)
    # # print(a, type(a))
    # b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # # print(a.shape)
    # # print(b)
    # # print(b.shape)
    # # print(a.dtype)
    # c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex)
    # print(c)

    # a = np.arange(1, 10, 0.5)
    # print(a)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    #绘图
    # mu = 0
    # sigma = 1
    # x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 51)
    # y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    # print(x.shape)
    # print('x = \n', x)
    # print(y.shape)
    # print('y = \n', y)


    # plt.figure(facecolor='w')
    # plt.plot(x, y, 'yo-', linewidth=1)
    # # plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=5)
    # plt.xlabel('X', fontsize=15)
    # plt.ylabel('Y', fontsize=15)
    # plt.title('高斯分布', fontsize=18)
    # plt.grid(True)
    # plt.show()


    # x = np.linspace(start=-2, stop=3, num=1001, dtype=np.float)
    # y_logit = np.log(1 + np.exp(-x)) / math.log(2)
    # y_boost = np.exp(-x)
    # y_01 = x < 0
    # y_hinge = 1.0 - x
    # y_hinge[y_hinge < 0] = 0
    # plt.figure(facecolor='w', figsize=(5, 5))
    # plt.plot(x, y_logit, 'r-', label='Logistic Loss')
    # plt.plot(x, y_01, 'g-', label='0/1 Loss')
    # plt.plot(x, y_hinge, 'b-', label='Hinge Loss')
    # plt.plot(x, y_boost, 'm--', label='Adaboost Loss')

    # x = np.arange(1, 0, -0.001)
    # y = (-3 * x * np.log(x) + np.exp(-(40 * (x - 1 / np.e)) ** 4) / 25) / 2
    # # plt.figure(facecolor='w', figsize=(10, 8))
    # plt.plot(y, x, 'r-')

    # t = np.linspace(0, 2*np.pi, 100)
    # x = 16 * np.sin(t) ** 3
    # y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)


    # t = np.linspace(0, 50, num=1000)
    # x = t*np.sin(t) + np.cos(t)
    # y = np.sin(t) - t*np.cos(t)

    # x = np.arange(0, 10, 0.1)
    # y = np.sin(x)
    # plt.bar(x, y, width=0.04, linewidth=0.2)
    # plt.plot(x, y, 'r--')
    # plt.xticks(rotation=-60)

    # plt.plot(x, y, 'r-')

    # t = 1000
    # a = np.zeros(10000)
    # for i in range(t):
    #     a += np.random.uniform(-5, 5, 10000)
    # a /= t
    # plt.hist(a, bins=30, color='g', alpha=0.5)

    u = np.linspace(-3, 3, 101)
    x, y = np.meshgrid(u, u)
    z = np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.Blue, linewidth=0.5)

    plt.grid()
    plt.legend()
    plt.show()

