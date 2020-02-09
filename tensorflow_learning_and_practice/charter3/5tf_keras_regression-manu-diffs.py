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
import pprint


housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

pprint.pprint(housing.data[0:5])
pprint.pprint(housing.target[0:5])

# train_test_split用于切分训练集和数据集
from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)   # 训练集测试机比例可以用test_size=?指定，默认0.25(3:1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)

# 归一化normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)
])

# fit函数作用
# 1.batch 遍历训练集 统计metric
#   1.1 自动求导
# 2.一次epoch结束 验证集metric

# metric的作用
metric = keras.metrics.MeanSquaredError()
print(metric([5.], [2.]))   # 9
print(metric([0.], [1.]))   # 1 -> 5 metric具有累加功能
print(metric.result())  # 5
metric.reset_states()   # 清空之前的数据（不想累加时使用）
print(metric([1.], [3.]))   # 4
# history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid), epochs=100, callbacks=callbacks)
metric.reset_states()

eporchs = 100
batch_size = 32
steps_per_eporch = len(x_train_scaled) // batch_size
optimizer = keras.optimizers.SGD()

def random_batch(x, y, batch_size=32):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]   # x, y都是numpy格式

for eporch in range(eporchs):
    metric.reset_states()
    for step in range(steps_per_eporch):
        x_batch, y_batch = random_batch(x_train_scaled, y_train, batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(y_batch, y_pred))
            metric(y_batch, y_pred)
        grads = tape.gradient(loss, model.variables)    # 对model里所有参数求梯度
        grads_and_vars = zip(grads, model.variables)    # 绑定梯度和参数
        optimizer.apply_gradients(grads_and_vars)
        print('\rEporch: ', eporch, 'train mse: ', metric.result().numpy(), end=" ")
    y_valid_pred = model(x_valid_scaled)
    valid_loss = tf.reduce_mean(keras.losses.mean_squared_error(y_valid_pred, y_valid))
    print('\t valid mse', valid_loss.numpy())

# def plot_learning_curves(history):
#     pd.DataFrame(history.history).plot(figsize=(8, 5))
#     plt.grid(True)
#     plt.gca().set_ylim(0, 1)
#     plt.show()
#
# plot_learning_curves(history)
# model.evaluate(x_test_scaled, y_test)