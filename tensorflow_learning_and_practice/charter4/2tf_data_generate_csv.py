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

output_dir = 'generate_csv'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def save_to_csv(output_dir, data, name_prefix, header=None, n_parts=10):
    path_format = os.path.join(output_dir, '{}_{:02d}.csv')
    filenames = []

    for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        with open(part_csv, 'wt', encoding='utf-8') as file:
            if header is not None:
                file.write(header+'\n')
            for row_index in row_indices:
                file.write(','.join([repr(col) for col in data[row_index]]))
                file.write('\n')
    return filenames

train_data = np.c_[x_train_scaled, y_train] # 按行进行merge
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]
header_cols = housing.feature_names + ['MidianHouseValue']
header_str = ','.join(header_cols)
train_filenames = save_to_csv(output_dir, train_data, 'train', header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data, 'valid', header_str, n_parts=10)
test_filenames = save_to_csv(output_dir, test_data, 'test', header_str, n_parts=10)

import pprint
print('train filename')
pprint.pprint(train_filenames)
print('valid filename')
pprint.pprint(valid_filenames)
print('test filename')
pprint.pprint(test_filenames)

print()
print()
print()
print()

# tensorflow 读取csv文件形成dataset
# 1.filenames -> dataset
# 2.read file by filenames -> datasets -> merge
# 3.parse merged csv
filename_dataset = tf.data.Dataset.list_files(train_filenames)
for filename in filename_dataset:
    print(filename)

n_readers = 5
dataset = filename_dataset.interleave(lambda filename : tf.data.TextLineDataset(filename).skip(1), cycle_length=n_readers)  # skip(1): 省略第一行即省略标题

for line in dataset.take(15):   # 只取前15个
    print(line.numpy())

# tf.io.decode_csv(str, record_defaults) str:csv文件一行记录的内容, record_defaults:定义字符串中各个数据的类型
# record_defaults 可以用[tf.constant(0, dtype=tf.int32), 0, np.nan, 'hello', 'tf.constant'] 来定义，即可以直接用值来定义，会自动识别为类型

def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields # 解析出来是float32类型
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y

def csv_reader_dataset(filenames, n_readers=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(lambda filename: tf.data.TextLineDataset(filename).skip(1), cycle_length=n_readers)
    dataset.shuffle(shuffle_buffer_size)    # 打乱数据
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads) # 类似于interleave, 不改变结构
    dataset = dataset.batch(batch_size)

    return dataset

# train_set = csv_reader_dataset(train_filenames, batch_size=3)
# for x_batch, y_batch in train_set.take(2):
#     print("x: ")
#     pprint.pprint(x_batch)
#     print("y: ")
#     pprint.pprint(y_batch)

batch_size = 32
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames, batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames, batch_size=batch_size)





model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[8]),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

history = model.fit(train_set, validation_data=valid_set, steps_per_epoch=11160 // batch_size, validation_steps = 3870 // batch_size, epochs=100, callbacks=callbacks)

model.evaluate(test_set, steps=5160 // batch_size)
