import networkx as nx
from itertools import product
import matplotlib.pyplot as plt

def net1(edge_number = 3):

    g = nx.Graph()
    inner_edges = [(x, (x+1)%edge_number) for x in range(edge_number)]
    g.add_edges_from(inner_edges)

    node_idx = edge_number
    for x, y in inner_edges:
        xx = node_idx
        yy = node_idx + 1
        node_idx += 2
        outer_nodes = [x, y, xx, yy]
        outer_edges = [(a, b) for a, b in product(outer_nodes, outer_nodes)
                       if a != b]
        g.add_edges_from(outer_edges)

    return g



if __name__ == '__main__':

    g = net1(4)
    nx.draw(g)
    plt.show()
    nx.write_gexf(g, r'./result/demo_4.gexf')
