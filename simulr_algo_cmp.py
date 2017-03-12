import algos
import mnets

import pandas as pd
import matplotlib.pyplot as plt

from validation import *
import os
import pickle

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

repeat_num = 100
path = 'temp/'
tempfile = path + 'algo_cmp.pickle'
algo_col = ['MLCD', 'OInfomap', 'ifp', 'Louvain', 'CPM']

if os.path.exists(tempfile):
    data = pickle.load(open(tempfile, 'rb'))
    repeat_num -= len(data)
else:
    data = []

input_cmd = mnets.lfr_cmd(n=500, k=10, maxk=50, mu=0.1, t1=2, on=50, om=2, minc=10, maxc=40)

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
    pickle.dump(data, open(tempfile, 'wb'))


os.remove(tempfile)
df_res = pd.DataFrame(data, columns=algo_col)

df_res.plot()
plt.show()
