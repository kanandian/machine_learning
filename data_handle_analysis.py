import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import animation

def f(x,y): #计算高度
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)


def data_handle():  #numpy and pandas
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


def animate(i):
    line.set_ydata(np.sin(x+i/10))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,


def data_show():    #matplotlib
    x = np.linspace(-3, 3, 50)
    y1 = 2*x+1
    y2 = x**2
    y3 = 3*x
    # plt.figure()    #第一个figure
    # plt.plot(x, y1)
    plt.figure(num=3, figsize=(8, 5))   #第二个figure
    l1, = plt.plot(x, y2, label='up')
    l2, = plt.plot(x, y3, color='red', linewidth=1.0, linestyle='--', label='down')

    #坐标轴
    plt.xlim((-1, 2))
    plt.ylim((-2, 3))
    plt.xlabel('x')
    plt.ylabel('y')

    x_ticks = np.linspace(-1, 2, 5)
    plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad\alpha$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
    plt.xticks(x_ticks)

    axis = plt.gca()    #get current axis
    axis.spines['right'].set_color('none')
    axis.spines['top'].set_color('none')
    # axis.xaxis.set_ticks_position('bottom') #默认底部线为x坐标轴
    # axis.yaxis.set_ticks_position('left ')   #默认左边线为y坐标轴
    axis.spines['bottom'].set_position(('data', 0)) #定位方式outward, axes
    axis.spines['left'].set_position(('data', 0))


    #图例 也可以事先指定label，然后直接调动plt.legend()不用参数
    plt.legend(handles=[l1, l2,], labels=['aaa', 'bbb'])
    plt.legend(handles=[l1,], labels=['aaa',])
    # plt.legend()

    #注解
    x0 = 1
    y0 = 3*x0
    plt.scatter(x0, y0, s=50, color='b')    #s:size
    plt.plot([x0, x0], [0, y0], 'k--', lw=2.5)  #k--:简写 表示黑色以及虚线;lw:line_width
    #method1
    plt.annotate(r'$3x=%s$'%y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))    #xy:注解定位;xycoords:xy的值以data为基准
    #method2
    # plt.text(-3.7, 3, r'$This\ is\ some\ text.\mu\ \sigma_i\ \alpha_t$', fontdict={'size':16, 'color':'g'})


    #tick能见度
    for label in axis.get_xticklabels()+axis.get_yticklabels():
        label.set_fontsize(12)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))    #alpha透明度

    #散点图
    X = np.random.normal(0, 1, 1024)    #正太分布(高斯分布)
    Y = np.random.normal(0, 1, 1024)
    T = np.arctan2(Y,X) #color for value
    plt.scatter(X, Y, s=75, c=T, alpha=0.5)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.xticks(())  #去掉ticks
    plt.yticks(())

    # 柱状图
    n = 12
    plt.figure()
    X = np.arange(n)
    Y1 = (1-X / float(n)*np.random.uniform(0.5, 1.0, n))
    Y2 = (1-X / float(n)*np.random.uniform(0.5, 1.0, n))

    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    for x, y in zip(X, Y1):
        plt.text(x, y+0.05, '%.2f'%y, ha='center', va='bottom') #ha:horizontal alignment;va:vetical alignment

    for x, y in zip(X, Y2):
        plt.text(x, -y-0.05, '%.2f'%y, ha='center', va='top') #ha:horizontal alignment;va:vetical alignment


    # 等高线图
    plt.figure()
    n = 256
    x = np.linspace(-3,3, n)
    y = np.linspace(-3,3, n)

    X,Y = np.meshgrid(x,y)
    plt.contourf(X,Y,f(X,Y),8, alpha=0.75, cmap=plt.cm.hot) #fill color
    C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidths=.5)  #画等高线

    plt.clabel(C, inline=True, fontsize=10)

    # # image
    # plt.figure()
    # a = np.linspace(0.3, 0.7, 9).reshape((3, 3))
    # plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower') #interpolation=[]显示效果;origin=['upper', 'lower']其中upper颜色反向
    # plt.colorbar(shrink=0.9)


    # 3D数据
    figure = plt.figure()
    axis = Axes3D(figure)
    X = np.arange(-4,4,0.25)
    Y = np.arange(-4,4,0.25)
    X, Y = np.meshgrid(X,Y)
    R = np.sqrt(X**2+Y**2)
    Z = np.sin(R)

    axis.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=plt.get_cmap('rainbow')) #rstride:row_stride:行跨度;cstride:column_stride;列跨度 分割线的紧凑程度
    axis.contourf(X, Y, Z, zdir='z', offset=-2, cmap='rainbow')    #zdir:从哪个轴压下去
    axis.set_zlim(-2, 2)

    # subplot多合一
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot([0,1], [0,2])

    plt.subplot(2,3,4)
    plt.plot([0, 1], [0, 3])

    plt.subplot(2, 3, 5)
    plt.plot([0, 1], [0, 2])

    plt.subplot(2, 3, 6)
    plt.plot([0, 1], [0, 2])



    # 分格显示(效果类似上节)
    # # method1:subplot2grid
    # plt.figure()
    # axis1 = plt.subplot2grid((3,3), (0,0), colspan=3, rowspan=1)
    # axis1.plot([1,2], [1,2])
    # axis1.set_xlabel('x')   #设置xlabel 之前所有设置函数前面都要加上set 例:plt.xlim()->axis.set_xlim()
    # axis1.set_ylabel('y')
    # axis1.set_title('axis1_title')
    # axis2 = plt.subplot2grid((3,3), (1,0), colspan=2)
    # axis3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
    # axis4 = plt.subplot2grid((3,3), (2,0))
    # axis5 = plt.subplot2grid((3,3), (2,1))


    # method2:gridspec
    plt.figure()
    gs = gridspec.GridSpec(3,3)
    axis1 = plt.subplot(gs[0,:])
    axis2 = plt.subplot(gs[1,:2])
    axis3 = plt.subplot(gs[1:,2])
    axis4 = plt.subplot(gs[-1,0])
    axis5 = plt.subplot(gs[2,1])


    # method3:easy to define structure(subplots)
    figure, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2,2, sharex=True, sharey=True)
    ax11.scatter([1,2], [1,2])


    # 图中图
    figure = plt.figure()
    x = [1,2,3,4,5,6,7]
    y = [1,3,4,2,5,8,6]

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = figure.add_axes([left, bottom, width, height])
    ax1.plot(x,y, 'r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
    ax2 = figure.add_axes([left, bottom, width, height])
    ax2.plot(y,x, 'g')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    left, bottom, width, height = 0.6, 0.2, 0.25, 0.25
    plt.axes([left, bottom, width, height])
    plt.plot(y[::-1], x, 'b')   #y[::-1]逆序
    plt.xlabel('x')
    plt.ylabel('y')


    #主次坐标轴
    x = np.arange(0,10,0.1)
    y1 = 0.05*x**2
    y2 = -1*y1

    figure, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x,y1,'g-')
    ax2.plot(x,y2,'b--')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y1')
    ax2.set_ylabel('Y2')



    # annimation
    figure, ax = plt.subplots()
    x = np.arange(0,2*np.pi, 0.01)
    line, = ax.plot(x,np.sin(x))

    ani = animation.FuncAnimation(fig=figure, func=animate, frames=100, init_func=init, interval=20, blit=False)

    plt.show()


if __name__ == '__main__':
    data_show()
