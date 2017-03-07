import algos
import mnets

import pandas as pd
import matplotlib.pyplot as plt

from validation import *

repeat_num = 50

algo_col = ['mlcd', 'oifp', 'ifp', 'M', 'CPM']

data = []

path = 'temp/'
input_cmd = mnets.lfr_cmd(n=500, k=18, maxk=35, mu=0, t1=2, on=20, om=2, minc=6, maxc=30)

for x in range(repeat_num):

    print('+++++++++ 第%3d/%3d次测试' % (x, repeat_num))
    # 生成 LFR 网络
    lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=3)
    networks = lfr_benchmark['networks']
    nodes = lfr_benchmark['nodes']

    mlcd = algos.Mlcd(networks)
    infomap = algos.Infomap(networks, path=path)
    louvain = algos.Louvain(networks)

    # (algo_func, parameters, option)
    algorithms = ((mlcd, None, True),
                  (infomap, '--overlapping --clu', False),
                  (infomap, '--clu', False),
                  (louvain, 'Modularity', False),
                  (louvain, 'CPM', False))

    d = []
    for a, p, opti in algorithms:

        r = a.run_algo(p)

        if opti is True:
            opti_node_coms, qoc = preprocess_node_community(r['node_coms'], nodes)
        else:
            opti_node_coms = r['node_coms']

        nmi = mni_olp_1(opti_node_coms,
                        lfr_benchmark['com2node'].values(),
                        len(nodes))

        d.append(nmi)

    data.append(d)

df_res = pd.DataFrame(data, columns=algo_col)
