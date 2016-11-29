from .edge import Edge

class TreeNode:
    Cnt = 0

    def __init__(self, info="node", simi=0.0, parent=None):
        self.parent = parent
        self.info = info
        self.children = []
        self.depth = 0
        self.simi = simi

        TreeNode.Cnt += 1

    def add_child(self, child):
        self.children.append(child)

    def add_children(self, children):
        self.children.extend(children)

    def set_parent(self, parent):
        self.parent = parent


class Dendrogram:
    """docstring for DendroGram"""

    def __init__(self, node_pairs_data, node_set):
        """
        系统树构造函数
        :param node_pairs_data: 原始系统树节点关系数据（已排好序）
        :param node_set: 原始系统树节点
        """
        self.__node_pairs_data = node_pairs_data
        self.__node_set = node_set

        # 保存系统树的所有子树的叶子集合，用于快速查找
        self.__leaves_info = dict()

        self.__pair_used = 0
        self.__pair_redu = 0

        # 系统树根节点
        self.__root = self.__generate_tree()



    def __generate_tree(self):
        """
        生成系统树图
        :return: 返回系统树图的根节点
        """

        # 初始迭代森林 [(TreeNode,(nodes..),...]
        forest = [(TreeNode(n, 1.0), {n}) for n in self.__node_set]

        for tree in forest:
            self.__leaves_info[tree[0]] = tuple(tree[1])

        for n1, n2, simi in self.__node_pairs_data:
            if abs(simi) < 0.0001 or len(forest) is 1:
                break

            # 相似对数据使用统计
            self.__pair_used += 1

            # 计算节点 n1和n2 所属的子树
            tree1 = Dendrogram.__find_tree(forest, n1)
            tree2 = Dendrogram.__find_tree(forest, n2)
            if tree1 is tree2:
                self.__pair_redu += 1
                continue

            if abs(tree1[0].simi - tree2[0].simi) < 0.0000001 and (
                    tree1[0].info == 'inner' or tree2[0].info == 'inner'):
                # 两棵树相似性相同，可以直接将两棵子树的孩子合并

                # 确定子树谁去谁留，作为叶节点的平凡子树是不能被移除的
                if tree1[0].info == 'inner':
                    remain_tree, left_tree = tree1, tree2
                else:
                    remain_tree, left_tree = tree2, tree1

                remain_tree[0].add_children(left_tree[0].children)
                remain_tree[1].update(left_tree[1])

                self.__leaves_info[remain_tree[0]] = tuple(remain_tree[1])
                self.__leaves_info.pop(left_tree[0])

                forest.remove(left_tree)

            else:
                # 两棵树相似性不同，需要添加新的节点进行融合
                new_tree_node = TreeNode(info="inner", simi=simi, parent=None)

                new_tree_node.add_child(tree1[0])
                new_tree_node.add_child(tree2[0])

                new_tree_node.depth = max(tree1[0].depth, tree2[0].depth) + 1
                tree1[0].set_parent(new_tree_node)
                tree2[0].set_parent(new_tree_node)

                new_nodes = tree1[1].union(tree2[1])

                forest.remove(tree1)
                forest.remove(tree2)
                forest.append((new_tree_node, new_nodes,))
                self.__leaves_info[new_tree_node] = tuple(new_nodes)



        if len(forest) is 1:

            forest[0][0].info = "root"
            return forest[0][0]
        else:

            # 不联通的情况
            new_tree_node = TreeNode(info="root", simi=0.0, parent=None)
            new_nodes = set()
            depth_max = 0
            for node in forest:
                new_tree_node.add_child(node[0])
                node[0].set_parent(new_tree_node)
                depth_max = max(depth_max, node[0].depth)
                new_nodes.update(node[1])
            new_tree_node.depth = depth_max + 1
            self.__leaves_info[new_tree_node] = tuple(new_nodes)
            return new_tree_node

    def __serialize(self, tree):
        if len(tree.children) is 0:
            return {'name': '%s' % tree.info, 'children': [], 'hight': tree.simi}
        else:
            children = []
            for child in tree.children:
                children.append(self.__serialize(child))
            return {'name': '%s' % tree.info, 'children': children, 'hight': tree.simi}

    def serialize(self, tree=None):
        return self.__serialize(tree=self.__root)

    def generate_community(self, cut_simi=0.0, least_com_num=0):

        # 系统树划分
        subtrees = Dendrogram.__cut_tree(self.__root, cut_simi)
        # 在该划分结果下的边社团
        covers = []
        for tree in subtrees:
            # 每一个社团
            one_com = self.__leaves_info[tree]
            if len(one_com) >= least_com_num:
                covers.append(tuple(one_com))
        return covers

    @classmethod
    def __find_tree(cls, forest, node):
        for tree in reversed(forest):
            if node in tree[1]:
                return tree
        return None

    @classmethod
    def __cut_tree(cls, tree, cut_simi):
        """
        系统树切割算法，按照切分相似度将相似度小于 cut_simi 的分支水平切割。
        算法的目的虽然是切割系统树，但是不会真正地破坏系统树结构，所以可以反复切割。
        :param tree: 系统树
        :param cut_simi: 切割相似度
        :return: 系统树水平切割后的森林
        """

        # 当前待切割的森林
        curr_forest = [tree]
        # 切割好的森林
        cut_well_trees = []

        is_continue_cut = True
        while is_continue_cut:
            is_continue_cut = False
            next_forest = []
            for subtree in curr_forest:
                # 分别处理根节点相似度 大于或小于 cut_simi 的情况
                if subtree.simi < cut_simi:
                    next_forest.extend(subtree.children)
                    is_continue_cut = True
                else:
                    cut_well_trees.append(subtree)

            curr_forest = next_forest

        return cut_well_trees



    @property
    def info(self):
        d = dict()
        d['depth'] = self.__root.depth
        d['root'] = self.__root
        d['leaf_num'] = len(self.__node_set)
        d['pair_num'] = len(self.__node_pairs_data)
        d['pair_used'] = self.__pair_used
        d['pair_redu'] = self.__pair_redu
        d['simi_min'] = self.__node_pairs_data[self.__pair_used-1][2]
        d['simi_max'] = self.__node_pairs_data[0][2]
        return d

