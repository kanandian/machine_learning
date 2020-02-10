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

# 读取csv文件成为pandas的DataFrame
train_file = '/Users/macpro/PycharmProjects/machine_learning/tensorflow_learning_and_practice/charter5/data/titanic/train.csv'
eval_file = '/Users/macpro/PycharmProjects/machine_learning/tensorflow_learning_and_practice/charter5/data/titanic/eval.csv'

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

print(train_df.head())  # head函数去除前几条数据，默认前5条
print(eval_df.head())

y_train = train_df.pop('survived')  # pop函数可以把相应的字段从数据集里去除，并返回该字段的值
y_eval = eval_df.pop('survived')

train_df.describe()  #显示统计信息

# 使用feature columns对数据做封装
# 将数据分为离散特征和连续特征
categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']   # 离散特征
numeric_columns = ['age', 'fare']   # 连续特征
feature_columns = []
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()   # unique函数获取一个属性所有可能的值
    print(categorical_column, vocab)
    feature_column = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(categorical_column, vocab))
    feature_columns.append(feature_column)

for numeric_column in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(numeric_column, dtype=tf.float32))


# 构建dataset
def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset

train_dataset = make_dataset(train_df, y_train, batch_size=5)


# 创建输出模型的文件夹
output_dir = 'baseline_model'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 创建estimator
baseline_estimator = tf.estimator.BaselineClassifier(model_dir=output_dir, n_classes=2)
baseline_estimator.train(input_fn=lambda : make_dataset(train_df, y_train, epochs=100))