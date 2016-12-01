import functools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlcd
import mnets

mu = 0.1
simi_algo = functools.partial(mlcd.linkpair_simi_2, alpha=0.5)

def process_algorithm(lcd_algo, layer_num):

    input_cmd = mnets.lfr_cmd(n=200, k=6, maxk=20, mu=mu, t1=2, on=4, om=2, minc=4, maxc=20)

    lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=layer_num)
    networks = lfr_benchmark['networks']
    lcd_algo.set_networks(networks)

    # �������Ƽ��㺯��������������������
    lcd_algo.set_linkpair_simi_algo(simi_algo)
    lcd_algo.cal_linkpair_similarity()

    # ϵͳ����Ϣ
    degram_info = lcd_algo.dendrogram.info

    qd = degram_info['pair_redu']/degram_info['pair_used']
    # ��������Ŀ�꺯������������������
    lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_max)
    result = lcd_algo.cal_optimization_community()

    cor_node_coms, qoc = mlcd.preprocess_node_community(result['node_coms'], lcd_algo.node_set)
    nmi = mlcd.mni_olp_1(cor_node_coms, lfr_benchmark['com2node'].values())

    return nmi, qoc, qd

repeat_num = 10

lcd_algo = mlcd.MNetworkLCD()

layer_num_seq = [1, 3, 5, 7 , 11, 13]
keys = ['L_%s' % l for l in layer_num_seq]


df_data = []


for r in range(repeat_num):


    for layer_num in layer_num_seq:

        print('+++++++++ ��%3d/%3d�β��ԣ����� layer=%3d ++++++++' % (r + 1, repeat_num, layer_num))

        nmi, qoc, qd = process_algorithm(lcd_algo, layer_num)

        df_data.append((layer_num, 's2', mu, nmi, qoc, qd))

col_name = ('layer', 's', 'mu', 'nmi', 'qoc', 'qd')

df_res = pd.DataFrame(data=df_data, columns=col_name)
