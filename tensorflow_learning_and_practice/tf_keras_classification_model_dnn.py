import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow import keras

import os

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

# normalization: x = (x - u) / std (其中u为训练集均值，std为训练集方差)
scaler = StandardScaler()
# x_train: [None, 28m 28] -> [None, 784] -> [None, 28, 28]
# fit_transform中fit的作用: 记录下数据集的均值和方差 (由于在做normalization时使用的均值和方差是训练集上的均值和方差，所以需要记录下来)
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)




print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

def show_single_image(img_arr):
    plt.imshow(img_arr, cmap='binary')
    plt.show()

def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows*n_cols < len(x_data)
    plt.figure(figsize=(n_cols*1.4, n_rows*1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols*row+col
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap='binary', interpolation='nearest')
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()


# show_single_image(x_train[0])
# class_names = ['T-shirt', 'Touser', 'Pullover', 'Dress', 'Coat', 'Sandal',
#                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# show_imgs(3, 5, x_train, y_train, class_names)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for i in range(20):
    model.add(keras.layers.Dense(100, activation='selu'))   #selu自带归一化
    # model.add(keras.layers.BatchNormalization())    #批归一化
    # 也可以将激活函数放在批归一化后面
    '''
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    '''
# model.add(keras.layers.Dropout(rate=0.5))
# model.add(keras.layers.AlphaDropout(rate=0.5))
# AlphaDropout特性:1.均值方差不变 2.归一化性质不变,因此可以和selu以及批归一化一起使用

model.add(keras.layers.Dense(10, activation='softmax')) #将向量变成概率分布

# reason for sparse: y->one_hot->向量
# 若y是一个向量类似[0.1, 0.2, 0.7]则使用categorical_crossentropy
# 若y是一个数通过one_hot转换成向量则使用sparse_categorical_crossentropy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()   #打出模型架构

# callback 回调函数
# TensorBoard, EarlyStopping, ModelCheckpoint
logdir = './dnn_callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, 'fashion_minst_model.h5')

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]
history = model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid), callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

model.evaluate(x_test_scaled, y_test)

# 发生的问题
# 1.参数多，训练补充恩（欠拟合）
# 2.梯度消失 -> 复合函数链式求导法则
#     批归一化可以缓解梯度消失
#     激活函数selu自带批归一化，因此也可以缓解梯度消失，且在训练时间及训练效果上都比批归一化好一点？