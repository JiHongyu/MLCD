import algos
import mnets

import pandas as pd
import matplotlib.pyplot as plt

from validation import *

repeat_num = 40

algo_col = ['mlcd', 'oifp', 'ifp']

data = []

path = 'temp/'
input_cmd = mnets.lfr_cmd(n=400, k=12, maxk=30, mu=0.1, t1=2, on=40, om=2, minc=6, maxc=30)

for x in range(repeat_num):

    print('+++++++++ 第%3d/%3d次测试' % (x, repeat_num))
    # 生成 LFR 网络
    lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=4)
    networks = lfr_benchmark['networks']
    nodes = lfr_benchmark['nodes']

    mlcd = algos.Mlcd(networks)
    infomap = algos.Infomap(networks, path=path)

    algorithms = (mlcd, infomap, infomap)

    d = []
    for a, s in zip(algorithms, range(len(algorithms))):

        if s is 1:
            r = a.run_algo('--overlapping --clu')
        elif s is 2:
            r = a.run_algo('--clu')
        else:
            r = a.run_algo()

        cor_node_coms, qoc = preprocess_node_community(r['node_coms'], nodes)
        nmi = mni_olp_1(cor_node_coms, lfr_benchmark['com2node'].values())

        d.append(nmi)

    data.append(d)

df_res = pd.DataFrame(data, columns=algo_col)
