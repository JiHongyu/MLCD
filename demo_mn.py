import matplotlib.pyplot as plt

import mlcd
import mnets


# gn :(n=128, k=16, maxk=16, mu=0.3, maxc=32, minc=32)
# (n=100, k=30, maxk=50, mu=0.2, t1=2.5, on=5, om=3)
input_cmd = mnets.lfr_cmd(n=128, k=16, maxk=16, mu=0.1, maxc=32, minc=32, on=0, om=2)
# 生成测试网络
print('生成测试网络数据')
lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=4)

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

# 4. 设置最优社团划算法
print('4. 设置最优社团划算法')
lcd_algo.set_objectfunc_algo(mlcd.objectfunc_1)

# 5. 寻找最优社团
print('5. 寻找最优社团')
result = lcd_algo.cal_optimization_community()

# 6. 获取系统树图
print('6. 获取系统树图')
lcd_algo.dump_dendrogram(path='.\\result\\')

# 7. 生成 gml 文件
print('7. 生成 gml 文件吧')
mnets.save_mn_benchmark(lfr_benchmark, '.\\result\\', link_coms=result['link_coms'])

# 8. 计算 Normalized mutual information
print('8. 计算 Normalized mutual information')
nmi = mlcd.mni_olp_1(result['node_coms'], lfr_benchmark['com2node'].values())
print(nmi)

plt.plot(result['curve'])
plt.show()
pass
