from .edge import Edge
from .objectivefunction import *
import heapq
from itertools import product

def com_simi(com1, com2):
    num1 = len(com1)
    num2 = len(com2)

    if num1 == 0 or num2 == 0:
        return 0
    inter_c = com1 & com2

    return len(inter_c)/min(num1, num2)


def community_fusion(networks, link_coms, node_coms):
    if len(link_coms) != len(node_coms):
        raise 'ERROR'
    coms_number = len(link_coms)

    # 初始化基本数据
    idx = 0
    coms_dict = dict()
    active_idx = set()

    for link_com, node_com in zip(link_coms, node_coms):
        coms_dict[idx] = (set(link_com), set(node_com))
        active_idx.add(idx)
        idx += 1

    # 初始化堆，因为 heapq 提供最小堆算法，所以我们要把相似度取最小值
    # heap for similarity
    # h=[(simi, idx, idy)...]
    h = []

    for x, y in product(coms_dict.keys(), coms_dict.keys()):
        if x < y:
            simi = com_simi(coms_dict[x][1], coms_dict[y][1])
            heapq.heappush(h, (-simi, x, y))

    best_density = -10000

    best_link_com = link_coms
    best_node_com = node_coms
    curve = [0]
    while len(active_idx) > 1:
        best_val = heapq.heappop(h)
        x = best_val[1]
        y = best_val[2]
        if x in active_idx and y in active_idx:

            new_link_com = coms_dict[x][0] | coms_dict[y][0]
            new_node_com = coms_dict[x][1] | coms_dict[y][1]

            delta_density = cuting_density(len(new_node_com), len(new_link_com)) - \
                            cuting_density(len(coms_dict[x][1]), len(coms_dict[x][0])) - \
                            cuting_density(len(coms_dict[y][1]), len(coms_dict[y][0]))

            if curve[-1] > 0 and delta_density + curve[-1] < 0:
                best_link_com = [coms_dict[x][0] for x in active_idx]
                best_node_com = [coms_dict[x][1] for x in active_idx]
            curve.append(delta_density + curve[-1])

            # 更新基本信息
            coms_dict[idx] = (new_link_com, new_node_com)
            active_idx.remove(x)
            active_idx.remove(y)
            # 更新堆数据
            for a in active_idx:
                simi = com_simi(new_node_com, coms_dict[a][1])
                heapq.heappush(h, (-simi, a, idx))
            active_idx.add(idx)
            idx += 1

    return curve, best_link_com, best_node_com













