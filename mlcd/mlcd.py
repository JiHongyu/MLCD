import json
import heapq

from collections import defaultdict
from .edge import Edge
from .mnetwork import MNetwork
from .dendrogram import Dendrogram
from .coms_fusion import *
from .similarity import *

class TreeNode:
    Cnt = 0
    def __init__(self, info="node", simi=0.0, parent=None):

        # 节点信息
        self.info = info
        # 父节点
        self.parent = parent
        # 子节点集
        self.left = None
        self.right= None
        # 节点深度
        self.depth = 0
        # 节点关联相似度
        self.simi = simi
        # 节点编号
        self.cnt = TreeNode.Cnt
        # 节点计数
        TreeNode.Cnt += 1
        # 子树叶子节点集
        self.leaves = set()
    def __hash__(self):
        return hash(self.info)



def convert_link2node_community(link_coms):
    node_coms = []
    for com in link_coms:
        com_node_set = set()
        for edge in com:
            com_node_set.update(edge.node())
        node_coms.append(tuple(com_node_set))
    return node_coms

def convert_edge2node(edge1, edge2):
    n11, n12 = edge1.node()
    n21, n22 = edge2.node()

    if n12 == n21:
        return n11, n12, n22
    else:
        return n11, n12, n21


#################################################
# 多网络连边检测算法主体

class MNetworkLCD:

    def __init__(self):

        self.mnetworks = None
        self.dendrogram = None
        self.func_curve = None

        self.linkpair = None
        self.link_set = None
        self.node_set = None

    def set_networks(self, networks):
        self.mnetworks = MNetwork(networks=networks)
        self.link_set = set(self.mnetworks.links())
        self.node_set = set(self.mnetworks.nodes())

    def set_linkpair_simi_algo(self, algo):

        self.mnetworks.set_linkpair_simi_algo(algo)

    def cal_dendrogram(self):

        self.linkpair = self.mnetworks.link_similarity_table()
        _links = self.mnetworks.links()
        self.dendrogram = Dendrogram(self.linkpair, _links)


    def cal_dendrogram_2(self):

        _link_list = list(self.link_set)

        # 初始化迭代数据

        # 系统树
        root = TreeNode(info=_link_list[0], simi=2.0)
        _link_cnt = len(_link_list) - 1

        # 最小堆，元素(-simi, choosed_edge, wait_edge)
        h = list()
        n1, n2 = _link_list[0].node()
        for n in self.mnetworks.project_nets.neighbors(n1):
            if n != n2:
                simi = linkpair_simi_2(
                    self.mnetworks.networksList, n2, n1, n)
                h.append((-simi, n2, n1, n))
        for n in self.mnetworks.project_nets.neighbors(n2):
            if n != n1:
                simi = linkpair_simi_2(
                    self.mnetworks.networksList, n1, n2, n)
                h.append((-simi, n1, n2, n))
        heapq.heapify(h)

        # 迭代计算
        while _link_cnt > 0 and len(h) > 0:
            _link_cnt -= 1
            # 提取堆顶元素
            rsimi, src, mid, dst = heapq.heappop(h)
            simi = -rsimi

            # 确定已插入边 e1 和带插入边 e2
            e1 = Edge(src, mid)
            e2 = Edge(dst, mid)

            # 无效连边
            if e2 in root.leaves:
                continue

            new_inner = TreeNode(info='inner', simi=simi)
            new_leaf = TreeNode(info=e2, simi=2.0, parent=new_inner)
            new_inner.left = new_leaf
            new_inner.leaves.add(e2)

            # 递归寻找插入点
            cur_node = root

            while cur_node.simi < simi:
                cur_node.leaves.add(e2)
                cur_node.depth += 1
                if e1 in cur_node.left.leaves:
                    cur_node = cur_node.left
                else:
                    cur_node = cur_node.right

            # 在二叉树里插入叶枝干
            parent = cur_node.parent

            new_inner.parent = parent
            new_inner.right = cur_node
            new_inner.leaves.update(cur_node.leaves)
            cur_node.parent = new_inner

            if parent == None:
                root = new_inner
            else:
                new_inner.depth = parent.depth - 1
                if parent.left == cur_node:
                    parent.left = new_inner
                else:
                    parent.right = new_inner

            for n in self.mnetworks.project_nets.neighbors(dst):
                if n != mid and n != src:
                    simi = linkpair_simi_2(
                        self.mnetworks.networksList, mid, dst, n)
                    heapq.heappush(h, (-simi, mid, dst, n))
        # 恢复系统树

        return root


    def set_objectfunc_algo(self, algo):
        self.mnetworks.set_objectfunc_algo(algo)

    def yield_communities(self, iterable=None):

        if iterable == None:
            cal_num = 20
            iterable = (x/cal_num for x in range(cal_num))

        for cut_simi in iterable:

            link_coms = self.dendrogram.generate_community(cut_simi=cut_simi, least_com_num=2)
            node_coms = convert_link2node_community(link_coms)
            cur_f = self.mnetworks.objectfunc(link_coms, node_coms)

            yield link_coms, node_coms, cur_f


    def cal_optimization_community(self, cal_num = 100, isOptimal=False):
        """利用划分密度进行树划分"""

        _curve = [0]*cal_num
        max_f = -100
        best_link_coms = None
        best_node_coms = None
        for depth in range(cal_num):

            cut_simi = (depth + 1) / cal_num
            # 获取当前深度下的划分结果
            link_coms = self.dendrogram.generate_community(cut_simi=cut_simi, least_com_num=1)

            node_coms = convert_link2node_community(link_coms)

            cur_f = self.mnetworks.objectfunc(link_coms, node_coms)

            if cur_f > max_f:
                max_f = cur_f
                best_link_coms, best_node_coms = link_coms, node_coms

            _curve[depth] = cur_f

        self.func_curve = _curve

        d = dict()
        # 这里待修改
        if isOptimal:
            d['link_coms'] = best_link_coms
            d['node_coms'] = best_node_coms
        else:
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

    def generate_community_2(self, root_tree, cut_simi, least_com_num):

        # 当前待切割的森林
        curr_forest = [root_tree]
        # 切割好的森林
        cut_well_trees = []

        is_continue_cut = True
        while is_continue_cut:
            is_continue_cut = False
            next_forest = []
            for subtree in curr_forest:
                # 分别处理根节点相似度 大于或小于 cut_simi 的情况
                if subtree.simi < cut_simi:
                    next_forest.append(subtree.left)
                    next_forest.append(subtree.right)
                    is_continue_cut = True
                else:
                    cut_well_trees.append(subtree)

            curr_forest = next_forest

        # 在该划分结果下的边社团
        covers = []
        for tree in cut_well_trees:
            # 每一个社团
            one_com = tree.leaves
            if len(one_com) >= least_com_num:
                covers.append(tuple(one_com))

        return covers

    def cal_optimization_community_2(self, cal_num = 100, isOptimal=False):
        """利用划分密度进行树划分"""

        root_tree = self.cal_dendrogram_2()

        _curve = [0]*cal_num
        max_f = -100
        best_link_coms = None
        best_node_coms = None
        for depth in range(cal_num):

            cut_simi = (depth + 1) / cal_num
            # 获取当前深度下的划分结果
            link_coms = self.generate_community_2(root_tree=root_tree,
                                                  cut_simi=cut_simi,
                                                  least_com_num=1)

            node_coms = convert_link2node_community(link_coms)

            cur_f = self.mnetworks.objectfunc(link_coms, node_coms)

            if cur_f > max_f:
                max_f = cur_f
                best_link_coms, best_node_coms = link_coms, node_coms

            _curve[depth] = cur_f

        self.func_curve = _curve

        d = dict()
        # 这里待修改
        if isOptimal:
            d['link_coms'] = best_link_coms
            d['node_coms'] = best_node_coms
        else:
            d['link_coms'] = best_link_coms
            d['node_coms'] = best_node_coms
        d['curve'] = _curve
        d['max_f'] = max_f
        return d



