import pickle

import matplotlib.pyplot as plt

import mlcd
import mnets


path = '.\\result\\'
# (n=100, k=30, maxk=50, mu=0.2, t1=2.5, on=5, om=3)
input_cmd = mnets.lfr_cmd(n=200, k=10, maxk=30, mu=0.0, t1=2, on=20, om=2, minc=5, maxc=40)
# 生成测试网络
print('生成测试网络数据')
lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=1)
mnets.save_mn_benchmark(lfr_benchmark, path=path)

networks = lfr_benchmark['networks']

# 多网络连边社团检测算法对象
lcd_algo = mlcd.MNetworkLCD()

# 1. 载入网络数据
print('1. 载入网络数据')
lcd_algo.set_networks(networks)

# 2. 设置相似性计算算法
print('2. 设置相似性计算算法')
lcd_algo.set_linkpair_simi_algo(mlcd.linkpair_simi_1)

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

# 7. 生成 gml 文件
print('7. 生成 gml 文件吧')
mnets.save_mn_benchmark(lfr_benchmark, path=path, link_coms=result['link_coms'])

# 8. 计算 Normalized mutual information
print('8. 计算 Normalized mutual information')

cor_node_coms, _ = mlcd.preprocess_node_community(result['node_coms'], lcd_algo.node_set)
nmi = mlcd.mni_olp_1(cor_node_coms, lfr_benchmark['com2node'].values())


redu_r = dgram_info['pair_redu']/dgram_info['pair_used']
print('NMI : %.4f'%nmi)
print('Redu: %.4f'%redu_r)

#pickle.dump(lcd_algo, open(path + 'mcld_algo.pickle', 'wb'))

plt.plot(result['curve'])
plt.show()
