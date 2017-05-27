import algos
import mnets
import networkx as nx
import copy
from validation import *

path = 'temp/'
tempfile = path + 'algo_cmp.pickle'
algo_col = ['MLCD', 'Infomap', 'Louvain']


g0 = nx.read_gexf(r'./result/data.gexf') # 输入数据文件路径
g = nx.Graph()
idx = 0
n2index = dict()
index2n = dict()
for n in g0.nodes():
    n2index[n] = idx
    index2n[idx] = n
    idx += 1

for n in g0.nodes():
    g.add_node(n2index[n])
for n1, n2 in g0.edges():
    g.add_edge(n2index[n1], n2index[n2])

networks = [g]
nodes = g.nodes()

mlcd = algos.Mlcd_v2(networks) # Mlcd_v2
infomap = algos.Infomap(networks, path=path)
louvain = algos.Louvain(networks)

algorithms = ((mlcd, None, True),
              (infomap, '--overlapping --clu', False),
              (louvain, 'Modularity', False))

d = []
cnt = 0
for a, p, opti in algorithms:

    r = a.run_algo(p)

    if opti is True:
        opti_node_coms, qoc = preprocess_node_community(r['node_coms'], nodes)
    else:
        opti_node_coms = r['node_coms']

    # 社团的结果
    print(opti_node_coms)

    # 保存社团结果

    name = algo_col[cnt]
    g_save = copy.deepcopy(g)

    if 'node_coms' in r:
        n2c = opti_node_coms
        x = 0
        for c in n2c:
            for n in c:
                g_save.node[n]['community'] = x
            x += 1
    if 'link_coms' in r:
        n2c = r['link_coms']
        x = 0
        for c in n2c:
            for e in c:
                n1, n2 = e.node()
                g_save.edge[n1][n2]['community'] = x
            x += 1
    cnt += 1
    nx.write_gexf(g_save, r'./result/%s.gexf'%name)


