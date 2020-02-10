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

# tfrecord 文件格式 -> tf.train.Example -> tf.train.Features -> {'key': tf.train.Features} 此处->表示 包含
# tf.train.Feature -> tf.train.ByteList/FloatList/Int64List 此处->表示 包含

favorite_book = [name.encode('utf-8') for name in ['machine learning', 'cc150']]
favorite_book_bytelist = tf.train.BytesList(value=favorite_book)
print(favorite_book_bytelist)
hours_floatlist = tf.train.FloatList(value = [15.5, 9.5, 7.0, 8.0])
age_int64list = tf.train.Int64List(value = [42])
features = tf.train.Features(
    feature={
        'favorite_books': tf.train.Feature(bytes_list=favorite_book_bytelist),
        'hours': tf.train.Feature(float_list=hours_floatlist),
        'age': tf.train.Feature(int64_list=age_int64list)
    }
)
# print(features)

example = tf.train.Example(features=features)
print(example)

# 序列化 压缩以减少文件大小
serialized_example = example.SerializeToString()
# print(serialized_example)

output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = 'test.tfrecords'
filepath = os.path.join(output_dir, filename)

# 写入文件
with tf.io.TFRecordWriter(filepath) as writer:
    for i in range(3):
        writer.write(serialized_example)


# 读取文件
dataset = tf.data.TFRecordDataset([filepath])
# for serialized_example_tensor in dataset:
#     print(serialized_example_tensor)

expected_features = {
    'favorite_books': tf.io.VarLenFeature(dtype=tf.string),
    'hours': tf.io.VarLenFeature(dtype=tf.float32),
    'age':tf.io.FixedLenFeature([], dtype=tf.int64)
}

for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
    # print(example)
    books = tf.sparse.to_dense(example['favorite_books'], default_value=b'')
    for book in books:
        print(book.numpy().decode('UTF-8'))


print()
print()
print()
print()
print()

# 压缩存储
filepath_zip = filepath+'.zip'
options = tf.io.TFRecordOptions(compression_type='GZIP')
with tf.io.TFRecordWriter(filepath_zip, options) as writer:
    for i in range(3):
        writer.write(serialized_example)

dataset = tf.data.TFRecordDataset([filepath_zip], compression_type='GZIP')

for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
    # print(example)
    books = tf.sparse.to_dense(example['favorite_books'], default_value=b'')
    for book in books:
        print(book.numpy().decode('UTF-8'))
