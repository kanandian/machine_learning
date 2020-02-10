import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow import keras

import os

dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)
for item in dataset:
    print(item)
print()
print()
print()


# 对dataset的常见操作
# 1.repeat eporch
# 2.get_batch
dataset = dataset.repeat(3).batch(7)
print(item)
for item in dataset:
     print(item)

print()
print()
print()
print()


# interleave
# case: 文件名dataset -> 文件内容dataset
dataset2 = dataset.interleave(lambda v : tf.data.Dataset.from_tensor_slices(v), #map_fn:做一个什么样的操作
                              cycle_length=5,   # 并行处理个数
                              block_length=5    # 从变化结果中每次去多少个元素出来
)
for item in dataset2:
    print(item)

print()
print()
print()
print()
print()

# 元祖作为参数
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)
for item_x, item_y in dataset3:
    print(item_x, item_y)

print()
print()
print()
print()

# 字典作为参数
dataset4 = tf.data.Dataset.from_tensor_slices({'feature': x, 'label': y})   # 参数是什么类型的，遍历时就是什么类型的
for item in dataset4:
    print(item['feature'], item['label'])



