import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlcd
import mnets

repeat_num = 10

lcd_algo = mlcd.MNetworkLCD()

nmi_data = []
com_qualify_data = []
linkpair_used_rate_data = []

mu_seq = np.linspace(0, 0.8, 10)

for mu in mu_seq:
    input_cmd = mnets.lfr_cmd(n=128, k=16, maxk=16, mu=mu, maxc=32, minc=32, on=0, om=0)

    t_nmi = []
    t_com_qualify = []
    t_linkpair_used_rate = []
    for r in range(repeat_num):
        print('参数 mu=%3f，第%3d次测试' % (mu, r))
        lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=2)
        networks = lfr_benchmark['networks']
        lcd_algo.set_networks(networks)

        # 设置相似计算函数，并计算连边相似性
        lcd_algo.set_linkpair_simi_algo(mlcd.linkpair_simi_1)
        lcd_algo.cal_linkpair_similarity()

        # 系统树信息
        degram_info = lcd_algo.dendrogram.info

        # 设置最有目标函数，并计算最优社团
        lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_max)
        result = lcd_algo.cal_optimization_community()

        cor_node_coms, com_quaty = mlcd.preprocess_node_community(result['node_coms'], lcd_algo.node_set)
        nmi = mlcd.mni_olp_1(cor_node_coms, lfr_benchmark['com2node'].values())

        #lcd_algo.dump_dendrogram(path='.\\result\\')
        #mnets.save_mn_benchmark(lfr_benchmark, '.\\result\\')

        t_nmi.append(nmi)
        t_com_qualify.append(com_quaty)
        t_linkpair_used_rate.append(degram_info['pair_redu']/degram_info['pair_used'])
        pass
    nmi_data.append(t_nmi)
    com_qualify_data.append(t_com_qualify)
    linkpair_used_rate_data.append(t_linkpair_used_rate)
    pass

df_nmi = pd.DataFrame(data=nmi_data).T
df_nmi.columns = ["%.3f" % x for x in mu_seq]

df_com_qualify = pd.DataFrame(data=com_qualify_data).T
df_com_qualify.columns = ["%.3f" % x for x in mu_seq]

df_linkpair_used_rate = pd.DataFrame(data=linkpair_used_rate_data).T
df_linkpair_used_rate.columns = ["%.3f" % x for x in mu_seq]

