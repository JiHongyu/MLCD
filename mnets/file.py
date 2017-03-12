import time
import itertools

import networkx as nx
import mlcd
import copy

def save_networks_container(networks,path:str,name=None):
    layer = len(networks)
    if name == None:
        name = 'MN_L'

    if not name.endswith('.gml'):
        name += '.gml'


def save_networks(mnetwork,path:str,name=None):
    pass

def save_sn_benchmark(benchmark, path:str, name=None, link_coms=None, node_coms=None):

    if name == None:
        name = 'sn_benchamrk'
    if not name.endswith('.gml'):
        name += '.gml'

    g = benchmark['network']
    node2com = benchmark['node2com']

    for n, c in node2com.items():
        _attr = {n: c}
        nx.set_node_attributes(g, 'benchmark', _attr)

    if link_coms is not None:
        label = 0
        for link_com in link_coms:
            _attr = {edge.node(): str(label) for edge in link_com}
            nx.set_edge_attributes(g, 'mlcd', _attr)
            label += 1

    nx.write_gml(g, path + name, stringizer=lambda x: str(x) if not isinstance(x, str) else x)

def save_mn_benchmark(benchmark, path:str,name=None, link_coms=None, node_coms=None):

    if name == None:
        name = 'mn_benchamrk'

    gs = copy.deepcopy(benchmark['networks'])
    node2com = benchmark['node2com']

    _layer = 1
    for g in gs:
        # for n, c in node2com.items():
        #     _attr = {n: c}
        #     nx.set_node_attributes(g, 'benchmark', _attr)
        #
        # edges = g.edges()
        # if link_coms is not None:
        #     label = 0
        #     for link_com in link_coms:
        #         _attr = {edge.node(): str(label) for edge in link_com if edge.node() in edges or edge.rnode() in edges}
        #         nx.set_edge_attributes(g, 'mlcd', _attr)
        #         label += 1

        nx.write_gexf(g, '%s_layer_%s.gexf' % (path + name, _layer))
        _layer += 1

def  save_sn_network(network, path:str, name=None, link_coms=None):

    g = copy.deepcopy(network)

    if name is None:
        name = 'sn_network'
    if not name.endswith('.gexf'):
        name += '.gexf'
    if link_coms is not None:
        label = 0
        for link_com in link_coms:
            for edge in link_com:

                x, y = edge.node()
                g[x][y]['mlcd'] = label
                # _attr = {edge.node(): str(label) for edge in link_com}
                # nx.set_edge_attributes(g, 'mlcd', _attr)
            label += 1
    nx.write_gexf(g, path=path + name)
