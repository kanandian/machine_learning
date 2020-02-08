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

# 自定义损失函数
def customized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))

# # layer的属性
# layer = tf.keras.layers.Dense(100, input_shape=(None, 5))
# print(layer(tf.zeros([10, 5])))
# print(layer.variables)  # layer中的变量
# print(layer.trainable_variables)    #layer中可训练的变量
# # print(help(layer))

#自定义layer 子类
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):   #用于构建所需要的参数，在自定义model是build函数和__init__合在一起
        # x * w + b   input_shape:[None, 输入的维度], output_shape:[None, units] -> w:[输入维度, units]
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.units), initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.units, ), initializer='zeros', trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):  # 完成正向运算
        return self.activation(x @ self.kernel + self.bias)

# 自定义layer lambda表达式
customized_softplus = keras.layers.Lambda(lambda x : tf.nn.softplus(x))
# print(customized_softplus([-10., 15., 5., 10.]))

model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu', input_shape=x_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus,    #等价于keras.layers.Dense(1, activation='softplus')
])

model.summary()
model.compile(loss=customized_mse, optimizer='adam', metrics=['mean_squared_error'])    #metrics:评估指标
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid), epochs=100, callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)
model.evaluate(x_test_scaled, y_test)