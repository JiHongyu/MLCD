import pickle

import matplotlib.pyplot as plt
import numpy as np

import mlcd
import mnets
from validation import *
import networkx as nx

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

path = '.\\result2\\'


networks = nx.read_gexf(r'./result/demo_3.gexf')

# 多网络连边社团检测算法对象
lcd_algo = mlcd.MNetworkLCD()

# 1. 载入网络数据
print('1. 载入网络数据')
lcd_algo.set_networks([networks])

# 2. 设置相似性计算算法
print('2. 设置相似性计算算法')
lcd_algo.set_linkpair_simi_algo(mlcd.linkpair_simi_2)

# 3. 计算连边相似性
print('3. 计算连边相似性')
lcd_algo.cal_linkpair_similarity()
dgram_info = lcd_algo.dendrogram.info

# 4. 设置最优社团划算法
print('4. 设置最优社团划算法')
lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_mean)

# 5. 寻找最优社团
print('5. 寻找最优社团')
cal_num = 100
result = lcd_algo.cal_optimization_community(cal_num=cal_num)

delta_density_info, best_link_com, best_node_com = \
    mlcd.community_fusion(None, result['link_coms'], result['node_coms'])
# 6. 获取系统树图
print('6. 获取系统树图')
lcd_algo.dump_dendrogram(path=path)

# 7. 生成 gml 文件
print('7. 生成可视化文件吧')

# 8. 计算 Normalized mutual information
print('8. 计算 Normalized mutual information')

cor_node_coms, _ = preprocess_node_community(result['node_coms'], lcd_algo.node_set)
lcd_link_coms = result['link_coms']
lcd_node_coms = result['node_coms']

c2, mlcd_link_coms, mlcd_node_coms = \
    mlcd.community_fusion(None, lcd_link_coms, lcd_node_coms)

label = 0
for link_com in lcd_link_coms:
    _attr = {edge.node(): str(label) for edge in link_com}
    nx.set_edge_attributes(networks, 'lcd', _attr)
    label += 1

label = 0
for link_com in mlcd_link_coms:
    _attr = {edge.node(): str(label) for edge in link_com}
    nx.set_edge_attributes(networks, 'mlcd', _attr)
    label += 1

nx.write_gexf(networks, r'./result/tempppp.gexf')

plt.plot([(x+1)/cal_num for x in range(cal_num)], result['curve'])
plt.xlabel('切分相似度门限 $s_t$')
plt.ylabel('划分密度')
plt.xlim(0, 1)
plt.savefig(r'./result2/density_curve.pdf')
plt.show()

d = np.zeros(len(delta_density_info) + 1)

yy = [x[0] for x in delta_density_info]
xx = [x[1] for x in delta_density_info]

d[0] = delta_density_info[0][0]
for x in range(len(delta_density_info)):
    d[x+1] = d[x] + delta_density_info[x][0]
plt.plot(d)
plt.show()

plt.plot(xx[1:], yy[1:])
plt.show()
