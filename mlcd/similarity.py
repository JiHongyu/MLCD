from itertools import product
import numpy as np
################################################
# 多网络连边相似性计算可选函数，后面的函数均需按照接口规范
# 函数接口：
# networks：多网络列表；
# src, mid, dst：依次是连边的起点，中间点，截止点；
# alpha：控制参数

def linkpair_simi_1(networks, src, mid, dst, alpha: float = 0.5):
    """ 计算节点相似性，原始方法，利用Jaccard度量方法
    输入: src, mid, dst: 边对标识符,alpha: 同层跨层控制系数
    输出: 节点相似性
    """
    # Case 1: complete src----mid----dst layers
    inlayer_src_neighbors = {src}
    inlayer_dst_neighbors = {dst}
    # Case 2: incomplete src--x--mid-----dst layers
    #          or        src-----mid--X--dst layers
    crslayer_src_neighbors = {src}
    crslayer_dst_neighbors = {dst}
    for graph in networks:
        if mid not in graph.nodes():
            continue

        if src in graph.neighbors(mid) and dst in graph.neighbors(mid):
            inlayer_src_neighbors.update(graph.neighbors(src))
            inlayer_dst_neighbors.update(graph.neighbors(dst))

        if src not in graph.neighbors(mid) and dst in graph.neighbors(mid):
            crslayer_src_neighbors.update(graph.neighbors(dst))

        if src in graph.neighbors(mid) and dst not in graph.neighbors(mid):
            crslayer_dst_neighbors.update(graph.neighbors(src))

    # Similarity
    # S1
    in_layer_simi = len(inlayer_src_neighbors & inlayer_dst_neighbors)\
                    / len(inlayer_src_neighbors | inlayer_dst_neighbors)
    # S2
    cross_layer_simi = len(crslayer_src_neighbors & crslayer_dst_neighbors)\
                       / len(crslayer_src_neighbors | crslayer_dst_neighbors)

    return (in_layer_simi + alpha * cross_layer_simi)/(1+alpha)


def linkpair_simi_2(networks, src, mid, dst, alpha: float = 0.5):

    inlayer_numerator = 0.0
    inlayer_denumerator = 0.0
    crslayer_numerator = 0.0
    crslayer_denumerator = 0.0

    num_of_nets = len(networks)

    neig = dict()

    for x in range(num_of_nets):
        neig[(x, src)] = set(networks[x].neighbors(src)) | {src}
        neig[(x, dst)] = set(networks[x].neighbors(dst)) | {dst}

    inlayer_cnt = 0
    crslayer_cnt = 0

    # 同层计算
    for x in range(num_of_nets):
        if src in networks[x].neighbors(mid) and dst in networks[x].neighbors(mid):
            inlayer_cnt += 1
            inlayer_numerator += len(neig[(x, src)] & neig[(x, dst)])
            inlayer_denumerator += np.sqrt(len(neig[(x, src)]) * len(neig[(x, dst)]))

    if inlayer_cnt > 1:
        inlayer_numerator /= inlayer_cnt
        inlayer_denumerator /= inlayer_cnt

    # 跨层计算
    for x, y in product(range(num_of_nets), range(num_of_nets)):
        if x != y and src in networks[x].neighbors(mid) and \
                      dst in networks[x].neighbors(mid):
            crslayer_cnt += 1
            crslayer_numerator += len(neig[(x, src)] & neig[(y, dst)])
            crslayer_denumerator += np.sqrt(len(neig[(x, src)]) * len(neig[(y, dst)]))

    if crslayer_cnt > 1:
        crslayer_numerator /= crslayer_cnt
        crslayer_denumerator /= crslayer_cnt

    simi_num = alpha*inlayer_numerator + (1-alpha)*crslayer_numerator
    simi_den = alpha*inlayer_denumerator + (1-alpha)*crslayer_denumerator

    return simi_num / simi_den if simi_den > 0.00001 else 0

