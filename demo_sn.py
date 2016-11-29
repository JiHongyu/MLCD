import matplotlib.pyplot as plt

import mlcd
import mnets


path = '.\\result\\'
# (n=100, k=30, maxk=50, mu=0.2, t1=2.5, on=5, om=3)
input_cmd = mnets.lfr_cmd(n=100, k=20, maxk=50, mu=0.05, t1=2, on=2, om=2)
# 生成测试网络
print('生成测试网络数据')
lfr_benchmark = mnets.lfr_sn_benchmark(command=input_cmd, is_new=True)
mnets.save_sn_benchmark(lfr_benchmark, path=path)

network = lfr_benchmark['network']

# 多网络连边社团检测算法对象
lcd_algo = mlcd.MNetworkLCD()

# 1. 载入网络数据
print('1. 载入网络数据')
lcd_algo.set_networks([network])

# 2. 设置相似性计算算法
print('2. 设置相似性计算算法')
lcd_algo.set_linkpair_simi_algo(mlcd.linkpair_simi_1)

# 3. 计算连边相似性
print('3. 计算连边相似性')
lcd_algo.cal_linkpair_similarity()
dgram_info = lcd_algo.dendrogram.info

# 4. 设置最优社团划算法
print('4. 设置最优社团划算法')
lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_max)

# 5. 寻找最优社团
print('5. 寻找最优社团')
result = lcd_algo.cal_optimization_community()

# 6. 获取系统树图
print('6. 获取系统树图')
lcd_algo.dump_dendrogram(path=path)

# 7. 生成 gml 文件
print('7. 生成 gml 文件吧')
mnets.save_sn_benchmark(lfr_benchmark, path=path, link_coms=result['link_coms'])

# 8. 计算 Normalized mutual information
print('8. 计算 Normalized mutual information')

cor_node_coms, _ = mlcd.preprocess_node_community(result['node_coms'], lcd_algo.node_set)
nmi = mlcd.mni_olp_1(cor_node_coms, lfr_benchmark['com2node'].values())


redu_r = dgram_info['pair_redu']/dgram_info['pair_used']
print('NMI : %.4f'%nmi)
print('Redu: %.4f'%redu_r)


plt.plot(result['curve'])
plt.show()
