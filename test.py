import pickle

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import mlcd
import mnets
from validation import *

path = '.\\result\\'
# (n=100, k=30, maxk=50, mu=0.2, t1=2.5, on=5, om=3)
input_cmd = mnets.lfr_cmd(n=1000, k=20, maxk=100, mu=0.00, t1=2, on=50, om=2, minc=20, maxc=100)
# 生成测试网络
print('生成测试网络数据')
lfr_benchmark = pickle.load(open(r'./result/lcd_mlcd.pickle', 'rb'))

networks = lfr_benchmark['networks']

# 多网络连边社团检测算法对象
lcd_algo = mlcd.MNetworkLCD()

# 1. 载入网络数据
print('1. 载入网络数据')
lcd_algo.set_networks(networks)

# 2. 设置相似性计算算法
print('2. 设置相似性计算算法')
lcd_algo.set_linkpair_simi_algo(mlcd.linkpair_simi_2)

# 3. 计算连边相似性
print('3. 计算连边相似性')
lcd_algo.cal_dendrogram()
dgram_info = lcd_algo.dendrogram.info

# 4. 设置最优社团划算法
print('4. 设置最优社团划算法')
lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_mean)

# 5. 寻找最优社团
print('5. 寻找最优社团')
result = lcd_algo.cal_optimization_community()

lcd_link_coms = result['link_coms']
lcd_node_coms = result['node_coms']

c2, mlcd_link_coms, mlcd_node_coms = \
    mlcd.community_fusion(None, lcd_link_coms, lcd_node_coms)

# 7. 生成 GEXF 文件
print('7. 生成可视化文件吧')
# mnets.save_sn_network(networks[0], path=path, name='lcd_result_lcd', link_coms=lcd_link_coms)
# mnets.save_sn_network(networks[0], path=path, name='lcd_result_mlcd', link_coms=mlcd_link_coms)
g = networks[0]
label = 0
for link_com in lcd_link_coms:
    _attr = {edge.node(): str(label) for edge in link_com}
    nx.set_edge_attributes(g, 'lcd', _attr)
    label += 1

label = 0
for link_com in mlcd_link_coms:
    _attr = {edge.node(): str(label) for edge in link_com}
    nx.set_edge_attributes(g, 'mlcd', _attr)
    label += 1

nx.write_gexf(g, r'./result/lcd_mlcd.gexf')

# 8. 计算 Normalized mutual information
print('8. 计算 Normalized mutual information')

lcd_v2_node_coms, _ = preprocess_node_community(lcd_node_coms, lcd_algo.node_set)
mlcd_v2_node_coms, _= preprocess_node_community(mlcd_node_coms, lcd_algo.node_set)

lcd_nmi = mni_olp_1(lcd_node_coms, lfr_benchmark['com2node'].values())
lcd_v2_nmi = mni_olp_1(lcd_v2_node_coms, lfr_benchmark['com2node'].values())
mlcd_nmi = mni_olp_1(mlcd_node_coms, lfr_benchmark['com2node'].values())
mlcd_v2_nmi = mni_olp_1(mlcd_v2_node_coms, lfr_benchmark['com2node'].values())

print('LCD NMI : %.4f, Community count %d' % (lcd_nmi, len(lcd_link_coms)))
print('LCD v2 NMI : %.4f, Community count %d' % (lcd_v2_nmi, len(lcd_v2_node_coms)))
print('MLCD NMI : %.4f, Community count %d' % (mlcd_nmi, len(mlcd_link_coms)))
print('MLCD v2 NMI : %.4f, Community count %d' % (mlcd_v2_nmi, len(mlcd_v2_node_coms)))

plt.plot(result['curve'])
plt.show()

plt.plot(c2)
plt.show()