import numpy as np

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

print(np.vstack((a, b)))    #上下合并
print(np.hstack((a, b)))    #左右合并


