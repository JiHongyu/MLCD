
################################################
# 多网络连边社团检测算法的最优社团选择的目标函数，目标函数是寻找极大值
# 函数接口：
# networks：多网络列表；
# link_coms：边社团
# node_coms：点社团

def objectfunc_1(networks, link_coms, node_coms):

    cur_f = 0
    all_link_num = 0
    num_of_net = len(networks)
    for link_com, node_com in zip(link_coms, node_coms):

        cur_com = len(link_com)

        min_com = (len(node_com)-1)*num_of_net
        max_com = 0.5 * len(node_com) * (len(node_com) - 1)*num_of_net

        density = (cur_com-min_com)/(max_com-min_com) if abs(max_com-min_com) > 0.0001 else 0

        cur_f += cur_com * density
        all_link_num += cur_com

    return cur_f/all_link_num
