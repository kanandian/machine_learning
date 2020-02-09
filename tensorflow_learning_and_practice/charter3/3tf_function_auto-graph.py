import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import pandas as pd
from tensorflow import keras
import os

# tf.function and autograph

# 方法一：使用tf.function(func)函数转换
def scaled_elu(z, scale=1.0, alpha=1.0):
    # z >= 0 ? scale * z : scale * alpha * tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))
print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3., -2.5])))

scaled_elu_tf = tf.function(scaled_elu) #tensorflow图函数 特点：块
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))

print(scaled_elu_tf is scaled_elu)
print(scaled_elu_tf.python_function is scaled_elu)  #可以从python_function属性中获取原来的python函数

# 方法二：annotation
# 1+1/2+...+1/2^n
@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total
print(converge_to_2(20))
print(converge_to_2.python_function)


# def display_tf_code(func):    # 转换后的函数代码
#     code = tf.autograph.to_code(func)
#     print(code)   # 展示代码


# tf.Variable无法在函数体内定义
# 在训练神经网络是用的更多的是tf.Variable,因此要在将函数转化成tf图函数之前将变量全部初始化

@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])   # 函数参数类型增加限定
def cube(z):
    return tf.pow(z,3)
try:
    print(cube(tf.constant([1., 2., 3.])))
except ValueError as ex:
    print(ex)
print(cube(tf.constant([1, 2, 3])))

# @tf.function: python_function -> graph函数
# get_concrete_function -> add input signature -> SavedModel
cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([None], tf.int32))
# cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([5], tf.int32))
# cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([1, 2, 3]))
print(cube_func_int32)
print(cube_func_int32.graph)    # 图定义
print(cube_func_int32.graph.get_operations())   # 图结构 操作
# cube_func_int32.graph.get_operation_by_name('x')  #根据名字获取operation
# cube_func_int32.graph.get_tensor_by_name('x:0')
cube_func_int32.graph.as_graph_def()





