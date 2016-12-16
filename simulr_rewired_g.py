import functools
import copy
import pandas as pd
import numpy as np

import mlcd
import mnets

path = '.\\result\\'
# (n=100, k=30, maxk=50, mu=0.2, t1=2.5, on=5, om=3)
input_cmd = mnets.lfr_cmd(n=1000, k=20, maxk=50, mu=0.1, t1=2, on=50, om=2, minc=10, maxc=50)


ori_lfr = mnets.lfr_sn_benchmark(input_cmd)

nxt_lfr = copy.deepcopy(ori_lfr)

mnets.rewire_benchmark(nxt_lfr['network'], nxt_lfr['com2node'], nxt_lfr['node2com'])


#mlcd.mni_olp_1()

mnets.save_sn_benchmark(ori_lfr, path=path, name='ori')
mnets.save_sn_benchmark(nxt_lfr, path=path, name='nxt')