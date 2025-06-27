# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
# 设置字体
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 32/3,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)

#df=pd.read_excel('1.xlsx',sheet_name='student')#可以通过sheet_name来指定读取的表单
# df = pd.read_excel('result_dim512_AWGN_snr.xlsx')#这个会直接默认读取到这个Excel的第一个表单
df = pd.read_excel('./WITT_channel_result.xlsx')#这个会直接默认读取到这个Excel的第一个表单
x = df.iloc[0, 2:8].values
y1 = df.iloc[1, 2:8].values
y2 = df.iloc[3, 2:8].values
y3 = df.iloc[5, 2:8].values
y4 = df.iloc[7, 2:8].values
y5 = df.iloc[9, 2:8].values
# x1 = df.iloc[0:3, 6].values
# y1 = df.iloc[0:3, 8].values
# x2 = df.iloc[3:6, 6].values
# y2 = df.iloc[3:6, 8].values
# x3 = df.iloc[6:9, 6].values
# y3 = df.iloc[6:9, 8].values

# y2 = df.iloc[10].values
# y3 = df.iloc[11].values


#开始画图
# plt.title('Result Analysis')
# plt.plot(x, y1, color='blue', label='所提算法', marker='o')
# plt.plot(x, y, color='black', marker='o')
plt.plot(x, y1, color='black', label='AWGN信道',marker='o')
plt.plot(x, y2, color='blue', label='瑞利信道', marker='x')
plt.plot(x, y3, color='red', label='莱斯信道，K=100', marker='s')
plt.plot(x, y4, color='orange', label='莱斯信道，K=1000', marker='s')
plt.plot(x, y5, color='yellow', label='莱斯信道，K=10000', marker='s')
# plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.legend() # 显示图例

plt.xlabel('信噪比',size = 32/3)
plt.ylabel('PSNR', size = 32/3)
plt.grid()
plt.show()
# plt.savefig('加权均方误差随信噪比变化',dpi=400)
#python 一个折线图绘制多个曲线
