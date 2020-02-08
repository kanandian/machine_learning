import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow import keras

import os

# constants
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
# index
# print(t)
# print(t[:, 1:])
# print(t[..., 1])    # 取第二列print(t[:, 1])

# op    # 基本上在keras-tensorflow中都有支持
print(t+10) #加法：每个元素都加10
print(tf.square(t)) #平方：每个元素都平方
print(t @ tf.transpose(t)) #乘转置

# numpy tensorflow 之间的转化
print(t.numpy())    # 转化成numpy的数组
print(np.square(t)) # tensorflow对象可以直接使用numpy函数加工
print(tf.constant(np.array([[1., 2., 3.], [4., 5., 6.]])))

# Scalars (0维)
t = tf.constant(2.789)
print(t.numpy())
print(t.shape)

# string
t = tf.constant('cafe')
print(t)
print(tf.strings.length(t)) # 字符串unicode编码长度
print(tf.strings.length(t, unit='UTF8_CHAR'))   # 字符串utf8编码长度
print(tf.strings.unicode_decode(t, "UTF8")) # 转化成utf8编码

# string array
t = tf.constant(['cafe', 'coffee', '咖啡'])
print(tf.strings.length(t, unit='UTF8_CHAR'))   # 字符串utf8编码长度
ragged_tensor = tf.strings.unicode_decode(t, "UTF8") # RaggedTensor：不等长tensor
print(ragged_tensor)

# raggedtensor
ragged_tensor = tf.ragged.constant([[11, 12], [21, 22, 23], [], [41]])
print(ragged_tensor)
print(ragged_tensor[1:2])   #左闭右开
#  ops on ragged tensor
ragged_tensor2 = tf.ragged.constant([[51, 52], [], [71], [82, 82]])
print(tf.concat([ragged_tensor, ragged_tensor2], axis=0))
print(tf.concat([ragged_tensor, ragged_tensor2], axis=1))
print(ragged_tensor.to_tensor())    #转化为普通的tensor,缺的部分用0补齐

# sparse tensor 记录数据不为0的数值的坐标及大小
sparse_tensor = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]], values=[1., 2., 3.], dense_shape=[3, 4])  # 其中indices表示有用数据的坐标, 必须是排好序的，否则会影响tf.sparse.to_dense(t)的使用，values表示有用数据的大小， dense_shape表示矩阵大小
print(sparse_tensor)
# 若indices没有排好序，则需调用
# sparse_tensor = tf.sparse.reorder(sparse_tensor)
print(tf.sparse.to_dense(sparse_tensor))    # 转化成普通tensor
# ops on sparse tensor
sparse_tensor2 = sparse_tensor*2
print(sparse_tensor2)

try:
    sparse_tensor3 = sparse_tensor+1    # sparse_tensor不能做加法
except TypeError as ex:
    print(ex)
tensor4 = tf.constant([[10., 2.], [30., 40], [50., 60.], [70., 80.]])
print(tf.sparse.sparse_dense_matmul(sparse_tensor, tensor4))    # sparse_tensor与普通tensor的乘法,得到一个普通的tensor



# Variables
v = tf.Variable([1., 2., 3.], [4., 5., 6.])
print(v)
print(v.value())
print(v.numpy())

# 变量的操作和常量差不多，而且会多一些操作
# 重新复制，是能使用assign函数，而不能使用'='
v.assign(2*v)
print(v)
v[0, 1].assign(42)
print(v)
v[1].assign([7., 8., 9.])
print(v)


