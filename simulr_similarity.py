import functools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlcd
import mnets


def process_algorithm(lcd_algo, simi_algo):

    # 设置相似计算函数，并计算连边相似性
    lcd_algo.set_linkpair_simi_algo(simi_algo)
    lcd_algo.cal_linkpair_similarity()

    # 系统树信息
    degram_info = lcd_algo.dendrogram.info

    qd = degram_info['pair_redu']/degram_info['pair_used']
    # 设置最有目标函数，并计算最优社团
    lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_max)
    result = lcd_algo.cal_optimization_community()

    cor_node_coms, qoc = mlcd.preprocess_node_community(result['node_coms'], lcd_algo.node_set)
    nmi = mlcd.mni_olp_1(cor_node_coms, lfr_benchmark['com2node'].values())

    return nmi, qoc, qd

repeat_num = 5

lcd_algo = mlcd.MNetworkLCD()

s1_alpha_seq = [0.1,0.3,0.5,0.7,0.9]
s2_alpha_seq = [0.1,0.3,0.5,0.7,0.9]

keys = ['s1_%.1f'%s1_a for s1_a in s1_alpha_seq]
for s2_a in s1_alpha_seq:
    keys.append('s2_%.1f'%s2_a)


mu_seq = np.linspace(0.1, 0.5, 5)


simi_algo_2 = functools.partial(mlcd.linkpair_simi_2, alpha=0.5)

nmi_df_data = []
qoc_df_data = []
qd_df_data = []

for r in range(repeat_num):

    t_res = {k:[] for k in keys}

    for mu in mu_seq:
        input_cmd = mnets.lfr_cmd(n=400, k=8, maxk=30, mu=mu, t1=2, on=8, om=2, minc=8, maxc=30)

        print('+++++++++ 第%3d/%3d次测试，参数 mu=%3f ++++++++' % (r+1, repeat_num, mu))

        lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=5)
        networks = lfr_benchmark['networks']
        lcd_algo.set_networks(networks)

        for s1_a in s1_alpha_seq:
            simi_algo_1 = functools.partial(mlcd.linkpair_simi_1, alpha=s1_a)
            _res = process_algorithm(lcd_algo, simi_algo_1)
            k = 's1_%.1f'%s1_a
            t_res[k].append(_res)

        for s2_a in s2_alpha_seq:
            simi_algo_2 = functools.partial(mlcd.linkpair_simi_2, alpha=s2_a)
            _res = process_algorithm(lcd_algo, simi_algo_2)
            k = 's2_%.1f' % s2_a
            t_res[k].append(_res)

    for k in keys:
        _t1 = [k]
        _t2 = [k]
        _t3 = [k]
        for nmi, qoc, qd in t_res[k]:
            _t1.append(nmi)
            _t2.append(qoc)
            _t3.append(qd)

        nmi_df_data.append(_t1)
        qoc_df_data.append(_t2)
        qd_df_data.append(_t3)






col_name = ['s']
for mu in mu_seq:
    col_name.append('%.3f'%mu)

df_nmi = pd.DataFrame(data=nmi_df_data, columns=col_name)
df_qoc = pd.DataFrame(data=qoc_df_data, columns=col_name)
df_qd = pd.DataFrame(data=qd_df_data, columns=col_name)