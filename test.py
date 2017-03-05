import pickle

import matplotlib.pyplot as plt
import networkx as nx
import mlcd
import mnets
from validation import *

path = '.\\result\\'


network = nx.read_gexf(path + 'ca_f.gexf')

# 多网络连边社团检测算法对象
lcd_algo = mlcd.MNetworkLCD()

# 1. 载入网络数据
print('1. 载入网络数据')
lcd_algo.set_networks([network])

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
result = lcd_algo.cal_optimization_community()

# 6. 获取系统树图
print('6. 获取系统树图')
lcd_algo.dump_dendrogram(path=path)

mnets.save_sn_network(network, path=path, link_coms=result['link_coms'])

plt.plot(result['curve'])
plt.show()
