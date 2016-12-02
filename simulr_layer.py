import functools
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlcd
import mnets

mu = 0.1
simi_algo = functools.partial(mlcd.linkpair_simi_2, alpha=0.5)

def process_algorithm(lcd_algo, layer_num):

    input_cmd = mnets.lfr_cmd(n=200, k=6, maxk=20, mu=mu, t1=2, on=20, om=2, minc=4, maxc=20)

    lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=layer_num)
    networks = lfr_benchmark['networks']
    lcd_algo.set_networks(networks)

    # 设置相似计算函数，并计算连边相似性
    lcd_algo.set_linkpair_simi_algo(simi_algo)
    lcd_algo.cal_linkpair_similarity()

    # 系统树信息
    degram_info = lcd_algo.dendrogram.info

    qd = degram_info['pair_redu']/degram_info['pair_used']
    # 设置最有目标函数，并计算最优社团
    lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_mean)
    result = lcd_algo.cal_optimization_community()

    cor_node_coms, qoc = mlcd.preprocess_node_community(result['node_coms'], lcd_algo.node_set)
    nmi = mlcd.mni_olp_1(cor_node_coms, lfr_benchmark['com2node'].values())

    return nmi, qoc, qd

repeat_num = 25

lcd_algo = mlcd.MNetworkLCD()

layer_num_seq = [1, 2, 3, 4, 5, 6]
keys = ['L_%s' % l for l in layer_num_seq]


df_data = []


for r in range(repeat_num):


    for layer_num in layer_num_seq:

        print('+++++++++ 第%3d/%3d次测试，参数 layer=%3d ++++++++' % (r + 1, repeat_num, layer_num))

        nmi, qoc, qd = process_algorithm(lcd_algo, layer_num)

        df_data.append((layer_num, 's2', mu, nmi, qoc, qd))

col_name = ('layer', 's', 'mu', 'nmi', 'qoc', 'qd')

df_res = pd.DataFrame(data=df_data, columns=col_name)

pickle.dump(df_res, open('layer_df.dump', 'wb'))