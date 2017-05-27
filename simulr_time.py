import algos
import mnets

import pandas as pd
import matplotlib.pyplot as plt
from mlcd.coms_fusion import community_fusion
from validation import *
import os
import pickle
import time

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

repeat_num = 100
path = 'temp/'
nmi_tempfile = path + 'algo_cmp.pickle'
time_tempfile = path + 'algo_cmp_2.pickle'
algo_col = ['MLCD_v2', 'MLCD']

if os.path.exists(nmi_tempfile):
    nmi_data = pickle.load(open(nmi_tempfile, 'rb'))
    time_data = pickle.load(open(time_tempfile, 'rb'))
    repeat_num -= len(nmi_data)
else:
    nmi_data = []
    time_data = []

input_cmd = mnets.lfr_cmd(n=1000, k=50, maxk=200, mu=0, t1=2, on=50, om=2, minc=80, maxc=200)

for x in range(repeat_num):

    print('+++++++++ 第%3d/%3d次测试' % (x, repeat_num))
    # 生成 LFR 网络
    lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=1)
    networks = lfr_benchmark['networks']
    nodes = lfr_benchmark['nodes']


    mlcd = algos.Mlcd(networks)
    mlcd_v2 = algos.Mlcd_v2(networks)

    algorithms = ((mlcd_v2, None, True), (mlcd, None, False))

    d_nmi = []
    d_time = []
    for a, p, opti in algorithms:

        start_time = time.time()
        r = a.run_algo(p)

        nc = r['node_coms']
        lc = r['link_coms']
        if opti:
            _, blc, bnc = community_fusion(None, lc, nc)
        else:
            blc, bnc = lc, nc
        # if opti is True:
        #     opti_node_coms, qoc = preprocess_node_community(, nodes)
        # else:
        #     opti_node_coms = r['node_coms']

        nmi = mni_olp_1(bnc,
                        lfr_benchmark['com2node'].values(),
                        len(nodes))
        end_time = time.time()

        d_time.append(end_time-start_time)
        d_nmi.append(nmi)

    time_data.append(d_time)
    nmi_data.append(d_nmi)
    pickle.dump(nmi_data, open(nmi_tempfile, 'wb'))
    pickle.dump(time_data, open(time_tempfile, 'wb'))

os.remove(nmi_tempfile)
os.remove(time_tempfile)

df_nmi = pd.DataFrame(nmi_data, columns=algo_col)
df_time = pd.DataFrame(time_data, columns=algo_col)
df_nmi.plot()
df_time.plot()

plt.show()
