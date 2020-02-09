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


def f(x):
    return 3.*x**2+2.*x-1
def approximate_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2. * eps)

print(approximate_derivative(f, 1))


def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)

def approximate_gradient(g, x1, x2, eps=1e-3):
    gd_x1 = approximate_derivative(lambda x : g(x, x2), x1, eps)
    gd_x2 = approximate_derivative(lambda x : g(x1, x), x2, eps)
    return gd_x1, gd_x2
print(approximate_gradient(g, 2., 3.))

# # 使用tensorflow一个一个求导
# x1 = tf.Variable(2.0)
# x2 = tf.Variable(3.0)
# with tf.GradientTape(persistent=True) as tape:  # 如果不设置persistent=True,在一次求导后tape会被自动消解，不能使用第二次，设置True后需要手动删除tape对象
#     z = g(x1, x2)
#
# dz_x1 = tape.gradient(z, x1)
# try:
#     dz_x2 = tape.gradient(z, x2)
# except RuntimeError as ex:
#     print(ex)
#
# del tape


# 使用tensorflow一起求导（梯度）
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x = tape.gradient(z, [x1, x2])
print(dz_x)

# 对常量求导
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x1)  # 当对常量求偏导是需要加这句话
    tape.watch(x2)
    z = g(x1, x2)

dz_x = tape.gradient(z, [x1, x2])
print(dz_x)


# 对个目标函数对同一个x求导
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3 * x
    z2 = x ** 2
print(tape.gradient([z1, z2], x))   # 得到z1对x的导数加z2对x的导数


# 求二阶导数
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grad, [[x1, x2]])
               for inner_grad in inner_grads]
del inner_tape
del outer_tape

# 手动实现梯度下降
learning_rate = 0.01
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    gd_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * gd_dx)
print(x)


# gradienttape结合keras optimizer
learning_rate = 0.01
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    gd_dx = tape.gradient(z, x)
    # x.assign_sub(learning_rate * gd_dx)
    optimizer.apply_gradients([(gd_dx, x)])
print(x)