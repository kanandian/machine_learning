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

if tf.test.is_built_with_gpu_support():
    # 设置gpu内存自增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
#     keras.layers.Dense(1)
# ])

# wide&deep模型 多输入 多输出
# 函数式API
# deep模型
input_width = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)

# 拼接wide模型的input和deep模型的output，这里假设wide模型和deep模型的input是一样的，input表示wide模型的输入，hidden2表示deep模型的输出
concat = keras.layers.concatenate([input_width, hidden2])
output = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(hidden2)

model = keras.models.Model(inputs=[input_width, input_deep], outputs=[output, output2])  #固化模型


model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

x_train_scaled_wide = x_train_scaled[:,:5]
x_train_scaled_deep = x_train_scaled[:,2:]
x_valid_scaled_wide = x_valid_scaled[:,:5]
x_valid_scaled_deep = x_valid_scaled[:,2:]
x_test_scaled_wide = x_test_scaled[:,:5]
x_test_scaled_deep = x_test_scaled[:,2:]
history = model.fit([x_train_scaled_wide, x_train_scaled_deep], [y_train, y_train], validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], [y_valid, y_valid]), epochs=100, callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)
model.evaluate([x_test_scaled_wide, x_test_scaled_deep], [y_test, y_test])