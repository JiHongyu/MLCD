import mlcd
import mnets

# 生成测试网络
print('生成测试网络数据')
gn_benchmark = mnets.gn_sn_benchmark()

mnets.save_sn_benchmark(gn_benchmark, '.\\')

network = gn_benchmark['network']

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

# 4. 设置最优社团划算法
print('4. 设置最优社团划算法')
lcd_algo.set_objectfunc_algo(mlcd.objectfunc_1)

# 5. 寻找最优社团
print('5. 寻找最优社团')
result = lcd_algo.cal_optimization_community()

# 6. 获取系统树图
print('6. 获取系统树图')
lcd_algo.dump_dendrogram(path='.\\')

# 7. 生成 gml 文件
print('7. 生成 gml 文件吧')
mnets.save_sn_benchmark(gn_benchmark, '.\\', link_coms=result['link_coms'])

# 8. 计算 Normalized mutual information
print('8. 计算 Normalized mutual information')
nmi = mlcd.mni_olp_1(result['node_coms'], gn_benchmark['com2node'].values())
print(nmi)


