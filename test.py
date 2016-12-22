import mnets
import algos
import tools

input_cmd = mnets.lfr_cmd(n=100, k=30, maxk=50, mu=0.2, t1=2.5, on=5, om=2)
# 生成测试网络
print('生成测试网络数据')
lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=3)

networks = lfr_benchmark['networks']

l = algos.Louvain(networks)

r = l.run_algo('Modularity')
pass