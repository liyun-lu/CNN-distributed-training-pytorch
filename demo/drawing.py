import matplotlib.pyplot as plt
import numpy as np

# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2. - 0.2, 1.03*height, '%s' % float(height))

# 构建数据
x_data = ['100', '500', '1000', '1500', '2000', '2500', '3000']
y_data = [9.044, 2.810, 2.121, 1.882, 1.646, 1.647, 1.672]
y_data2 = [9.056, 2.781, 2.071, 1.843, 1.561, 1.619, 1.618]
bar_width = 0.4
# 将X轴数据改为使用range(len(x_data), 就是0、1、2...
a = plt.bar(x=range(len(x_data)), height=y_data, label='TCP/IP',
            color='#0072BC', alpha=0.8, width=bar_width)
# 将X轴数据改为使用np.arange(len(x_data))+bar_width
# 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了
b = plt.bar(x=np.arange(len(x_data)) + bar_width, height=y_data2,
        label='Shared File', color='#ED1C24', alpha=0.8, width=bar_width)
# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
autolabel(a)
autolabel(b)
# 为两条坐标轴设置名称
plt.xlabel("Batch_size")
plt.ylabel("Time/mins")
x = np.arange(7)
plt.xticks(x+bar_width/2, x_data)
# 显示图例
plt.legend()
plt.show()

