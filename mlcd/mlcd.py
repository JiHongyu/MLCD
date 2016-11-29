import json

from .mnetwork import MNetwork
from .dendrogram import Dendrogram



def convert_link2node_community(link_coms):
    node_coms = []
    for com in link_coms:
        com_node_set = set()
        for edge in com:
            com_node_set.update(edge.node())
        node_coms.append(tuple(com_node_set))
    return node_coms

def preprocess_node_community(node_coms, nodes):

    fixed_node_coms = []
    fixed_nodes = set()

    for com in node_coms:
        if len(com) > 2:
            fixed_nodes.update(com)
            fixed_node_coms.append(com)

    diff_nodes = nodes - fixed_nodes

    for link in diff_nodes:
        fixed_node_coms.append((link,))

    return fixed_node_coms, 1-len(diff_nodes)/len(nodes)

#################################################
# 多网络连边检测算法主体

class MNetworkLCD:

    def __init__(self):

        self.mnetworks = None
        self.dendrogram = None
        self.func_curve = None

        self.link_set = None
        self.node_set = None

    def set_networks(self,networks):
        self.mnetworks = MNetwork(networks=networks)
        self.link_set = set(self.mnetworks.links())
        self.node_set = set(self.mnetworks.nodes())

    def set_linkpair_simi_algo(self, algo):

        self.mnetworks.set_linkpair_simi_algo(algo)

    def cal_linkpair_similarity(self):

        _linkpair = self.mnetworks.link_similarity_table()
        _links = self.mnetworks.links()
        self.dendrogram = Dendrogram(_linkpair, _links)

    def set_objectfunc_algo(self, algo):
        self.mnetworks.set_objectfunc_algo(algo)

    def cal_optimization_community(self, cal_num = 100):
        """利用划分密度进行树划分"""

        _curve = [0]*cal_num
        max_f = 0
        best_link_coms = None
        best_node_coms = None
        for depth in range(cal_num):

            cut_simi = (depth + 1) / cal_num
            # 获取当前深度下的划分结果
            link_coms = self.dendrogram.generate_community(cut_simi=cut_simi, least_com_num=1)

            node_coms = convert_link2node_community(link_coms)

            #node_coms, used_rate = preprocess_node_community(node_coms, self.node_set)

            cur_f = self.mnetworks.objectfunc(link_coms, node_coms)

            if cur_f > max_f:
                max_f = cur_f
                best_link_coms, best_node_coms = link_coms, node_coms

            _curve[depth] = cur_f

        self.func_curve = _curve

        d = dict()
        d['link_coms'] = best_link_coms
        d['node_coms'] = best_node_coms
        d['curve'] = _curve
        d['max_f'] = max_f
        return d

    def dump_dendrogram(self, path : str, name=None):

        if name == None:
            name = 'tree_data.txt'

        try:
            tree_ser = self.dendrogram.serialize()
            json.dump(tree_ser, open(path + name, 'w'))

        except:
            print(" Dump Dendrogram ERROE")
