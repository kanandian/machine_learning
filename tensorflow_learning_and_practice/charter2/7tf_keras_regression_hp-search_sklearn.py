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

callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]
# 使用RandomizedSearchCV(sklearn里的参数)进行超参数搜索
# 1.转化为sklean的model
# 2.定义超参数集合
# 3.使用使用RandomizedSearchCV搜索参数


# 1.转化为sklean的model,使用KerasRegressor(用于回归模型)或KerasClassifier(用于分类模型)
def build_model(hidden_layers=1, layer_size=30, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation='relu', input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers-1):
        model.add(keras.layers.Dense(layer_size, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)
# history = sklearn_model.fit(x_train_scaled, y_train, epochs=100, validation_data=(x_valid_scaled, y_valid), callbacks=callbacks)

# 2.定义超参数集合
from scipy.stats import reciprocal
# f(x) = 1/(x*log(b/a)) a <= x <= b
param_distribution = {
    'hidden_layers':[4],
    'layer_size':[84,85,86],
    'learning_rate':[1e-4, 3e-4, 1e-3],
    # 'learning_rate':reciprocal(1e-4, 1e-2),
}
# reciprocal.rvs(1e-4, 1e-2, size=3)

# 3.使用使用RandomizedSearchCV搜索参数
from sklearn.model_selection import RandomizedSearchCV
random_search_cv = RandomizedSearchCV(sklearn_model, param_distribution, cv=4, n_iter=1, n_jobs=1)   #n_iter表示从param_distribution中sample出多少个参数集合,n_jobs表示有多少任务并行处理
random_search_cv.fit(x_train_scaled, y_train, epochs=100, validation_data=(x_valid_scaled, y_valid), callbacks=callbacks)
# 在进行超参数搜索时,RandomizedSearchCV会使用交叉验证即cross_validation:将训练集分为n分，其中n-1分用于训练，1分用于验证，可以用cv=n进行指定，超参数搜索完后会再用得到的参数使用全部训练集进行训练
# 因此，训练数据集会比原来的训练集小，超参数搜索完后，训练集数据大小和原来的训练集相同

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

# plot_learning_curves(history)
# model.evaluate(x_test_scaled, y_test) #sklearn_model没有evaluate函数
print(random_search_cv.best_params_)
print(random_search_cv.best_score_)
print(random_search_cv.best_estimator_) # 获取最好的model

model = random_search_cv.best_estimator_.model
model.evaluate(x_test_scaled, y_test)