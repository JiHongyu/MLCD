import matplotlib.pyplot as plt
import networkx as nx

import mlcd
import mnets
from validation import *

g1 = nx.read_gexf(r'./result/ca_mlcd_1.gexf')
g2 = nx.read_gexf(r'./result/ca_mlcd_1.gexf')
g3 = nx.read_gexf(r'./result/ca_mlcd_1.gexf')
networks = [g1, g2, g3]

g0 = nx.Graph()
for g in networks:
    g0.add_nodes_from(g.nodes())
    g0.add_edges_from(g.edges())


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

# 4. 设置最优社团划算法
print('4. 设置最优社团划算法')
lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_max)

# 5. 寻找最优社团
print('5. 寻找最优社团')
result = lcd_algo.cal_optimization_community()

# 6. 获取系统树图
print('6. 获取系统树图')
lcd_algo.dump_dendrogram(path='.\\result\\')

lcd_link_coms = result['link_coms']
lcd_node_coms = result['node_coms']

c2, mlcd_link_coms, mlcd_node_coms = \
    mlcd.community_fusion(None, lcd_link_coms, lcd_node_coms)

mnets.save_sn_network(g0, path='.\\result\\', name='ca_mlcd', link_coms=mlcd_link_coms)
# plt.plot(result['curve'])
plt.plot(c2)
plt.show()
pass
