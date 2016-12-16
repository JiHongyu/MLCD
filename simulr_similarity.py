import functools
import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

import mlcd
import mnets


def process_algorithm(lcd_algo, simi_algo):

    print('计算相似性')
    # 设置相似计算函数，并计算连边相似性
    lcd_algo.set_linkpair_simi_algo(simi_algo)
    lcd_algo.cal_linkpair_similarity()

    # 系统树信息
    degram_info = lcd_algo.dendrogram.info

    qd = degram_info['pair_redu']/degram_info['pair_used']
    print('计算最优社团')
    # 设置最有目标函数，并计算最优社团
    lcd_algo.set_objectfunc_algo(mlcd.objectfunc_by_max)
    result = lcd_algo.cal_optimization_community()

    cor_node_coms, qoc = mlcd.preprocess_node_community(result['node_coms'], lcd_algo.node_set)
    nmi = mlcd.mni_olp_1(cor_node_coms, lfr_benchmark['com2node'].values())

    return nmi, qoc, qd

repeat_num = 40
num_of_layer = 3

lcd_algo = mlcd.MNetworkLCD()

s1_alpha_seq = [0.5, 0.9]
s2_alpha_seq = [0.5, 0.9]



mu_seq = np.linspace(0, 0.2, 3)


df_data = []

error_num = 0

for r in range(repeat_num):

    for mu in mu_seq:
        input_cmd = mnets.lfr_cmd(n=800, k=20, maxk=50, mu=mu, t1=2, on=40, om=2, minc=10, maxc=50)

        print('+++++++++ 第%3d/%3d次测试，参数 mu=%.3f ++++++++' % (r+1, repeat_num, mu))

        lfr_benchmark = mnets.lfr_mn_benchmark(input_cmd, num_of_layer=num_of_layer)
        networks = lfr_benchmark['networks']
        lcd_algo.set_networks(networks)

        try:
            for s1_a in s1_alpha_seq:
                simi_algo_1 = functools.partial(mlcd.linkpair_simi_1, alpha=s1_a)
                nmi, qoc, qd = process_algorithm(lcd_algo, simi_algo_1)
                df_data.append((num_of_layer, mu, 's1', s1_a, nmi, qoc, qd))

            for s2_a in s2_alpha_seq:
                simi_algo_2 = functools.partial(mlcd.linkpair_simi_2, alpha=s2_a)
                nmi, qoc, qd = process_algorithm(lcd_algo, simi_algo_2)
                df_data.append((num_of_layer, mu, 's2', s2_a, nmi, qoc, qd))

        except:
            print('There is a error')
            error_num += 1
            continue

col_name = ('layer', 'mu', 's', 'alpha', 'nmi', 'qoc', 'qd')


df_res = pd.DataFrame(data=df_data, columns=col_name)

pickle.dump(df_res, open('simi_df.dump','wb'))