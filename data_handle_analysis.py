import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

array = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((10, 10), dtype=float)
random = np.random.random((2,4))
a = np.arange(0, 12, 2).reshape((2, 3))
b = np.linspace(1, 10, 6).reshape(2, 3)
c = b-a
c = b+a
c = a*b #逐个相乘
# c = np.dot(a,b) #矩阵乘法 c = a.dot(b)
c = 10*np.sin(a)
print(c<0)
sum = np.sum(random) #np.min(random)    np.max(random)
sum = np.sum(random, axis=0)    #0:统计行 1:统计列

a = np.random.randint(1, 5, size=(3, 4))
print(np.argmin(a)) #最小值的索引 np.argmax(a)
print(a.mean()) #np.mean(a)一样
print(np.cumsum(a)) #累加 a(i+1)=a(i)+n(i)
print(np.diff(a))   #累差a(i) = a(i+1)-a(i) shape:(m, n)->(m, n-1)
print(np.nonzero)   #使用两个array表示非零值的索引，第一个array表示横坐标，第二个表示纵坐标
print(np.sort(a))   #逐行排序
print(a.T)  #转置 np.transpose(a)
print(np.clip(a, 3, 4)) #小于3的等于三，大于4的等于4


"""
    ndarray可以向列表一样访问（使用下标）
"""
print(a[2][1])
print(a[1, 2])
print(a[1, 1:3])

for row in a:   #迭代行
    print(row)
for column in a.T: #迭代列
    print(column)

print(a.flatten())


"""
    ndarray 合并
"""

# print(np.vstack((a, b)))    #上下合并
# print(np.hstack((a, b)))    #左右合并
#或使用
# print(np.concatenate(a,b,a), axis=0)
print(a[:,np.newaxis])  #新增一个维度 可用reshape实现


"""adarray 分割"""
print(np.split(a, 2, axis=1))   #等项分割 分成两个ndarray
#也可以用np.vsplit np.hsplit
print(np.array_split(a,3,axis=1))   #不等项分割


"""赋值 复制"""
b = a
print(a is b)   #True
b = a.copy()    #deep copy
print(a is b)   #False


"""pandas"""
s = pd.Series([1,2,3,4,5,np.nan,6,7])
dates = pd.date_range('20200101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['a','b','c','d'])
print(df.dtypes, df.index, df.columns, df.values)
print(df.describe())
df.sort_index(axis=1, ascending=False)  #对index进行排序
df.sort_values(by='a')

"""选择数据"""
print(df['a'])  #df.a 效果相同
print(df[0:3])
print(df['20200101':'20200104'])
#select by label
print(df.loc['20200101', ['a','b']])
#select by position
print(df.iloc[[1,2,4],0:2])
#mixed selection
# print(df.ix[])    #被弃用
print(df[df.a>0])


"""赋值 直接赋值"""
df.a[df.b<0] = 0
df['e'] = 0 #新增一列
df['e'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20200101', periods=6)) #新增一列 不能缺少index


"""丢失数据处理NaN"""
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df.dropna(axis=0,how='any'))  #how={'any', 'all'} 默认any
print(df.fillna(value=0))
print(df.isnull())
print(np.any(df.isnull())==True)


"""导入导出(读入写出)"""
data = pd.read_csv('PM2.5Predection/PM2.5_DATA/train.csv')
data.to_pickle('train.pickle')


"""合并"""
# res = pd.concat([df1, df2, df3], axis=0, ignore_index=True, join='inner') #join={'outer', 'inner'} 默认outer
# res = pd.concat([df1, df2, df3], axis=0, ignore_index=True, join_axes=[df1.index])
# res = pd1.append(df2, ignore_index=True)
# res = pd1.append([df2, df3], ignore_index=True)
# s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
# res.dp.append(s1, ignore_index=True)


# res = pd.merge(pd1, pd2, on='key', indicator=True)
# res = pd.merge(pd1, pd2, on=['key1', 'key2' how='inner'], indicator='indicator_column') #默认inner how=['left', 'right', 'outer', 'inner']
# res = pd.merge(pd1, pd2, pd1_index=True, pd2_index=True, how='outer') #通过index merge
# res = pd.merge(pd1, pd2, on='key', suffixes=['_pd1', '_pd2'], how='inner') #用于分辨不同表中意义不同的同名属性


"""数据可视化"""
#series
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
# data.plot()
# plt.show()

#data_frame
data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
data = data.cumsum()
# data.plot()
# plt.show()

"""
    plot methods:
    'bar', 'hist', 'box', 'kde', 'area', 'scatter', 'hexbin', 'pie'
"""
class1 = data.plot.scatter(x='A', y='B', color='DarkBlue', label='class1')
class1 = data.plot.scatter(x='A', y='C', color='DarkGreen', label='class2', ax=class1)
plt.show()

