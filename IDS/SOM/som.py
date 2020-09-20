from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from minisom import MiniSom


def classify(som, data, win_map):
    default_class = np.sum(list(win_map.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in win_map:
            result.append(win_map[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

in_data = pd.read_csv('E:\\论文\\小论文两篇\\IDS实验3\\train_data_scaler.csv', sep='\t', header=None)
X = np.array(in_data.iloc[:, :69])
y = np.array(in_data.iloc[:, 69:70], dtype=np.int32)

y = y.reshape(y.shape[0]*y.shape[1], )

# 划分训练集、测试集  7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

N = X_train.shape[0]  # 样本数量
M = X_train.shape[1]  # 维度/特征数量

print('M=%d' % M)
print('N=%d' % N)

'''
设置超参数
'''
size = np.math.ceil(np.sqrt(5 * np.sqrt(N)))  # 经验公式：决定输出层尺寸
print("训练样本个数:{}  测试样本个数:{}".format(N, X_test.shape[0]))
print("输出网格最佳边长为:", size)

max_iter = 200

# Initialization and training
som = MiniSom(size, size, M, sigma=3, learning_rate=0.5,
              neighborhood_function='bubble')

'''
初始化权值，有2个API
'''
# som.random_weights_init(X_train)
som.pca_weights_init(X_train)

som.train_batch(X_train, max_iter, verbose=False)

# som.train_random(X_train, max_iter, verbose=False)


win_map = som.labels_map(X_train, y_train)
y_pred = classify(som, X_test, win_map)
print(classification_report(y_test, np.array(y_pred)))

heatmap = som.distance_map()  # 生成U-Matrix
plt.imshow(heatmap, cmap='bone_r')  # miniSom案例中用的pcolor函数,需要调整坐标
plt.colorbar()
plt.show()
