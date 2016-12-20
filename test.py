import mnets
import algos

path = 'temp/'
input_cmd = mnets.lfr_cmd(n=1000, k=30, maxk=50, mu=0, t1=2.5, on=5, om=2)
# 生成测试网络
print('生成测试网络数据')
lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=1)

networks = lfr_benchmark['networks']

infomap = algos.Infomap(networks, path=path)

infomap.run_algo()