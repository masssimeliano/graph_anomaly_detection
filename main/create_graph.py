# Script to create graph from dataset

import networkx as nx
import matplotlib.pyplot as plt

def create(data):
    graph  = nx.Graph()

    # Dinamically for illustration
    graph_size = 10000

    row = 0
    while (row < 11944 and graph_size != 0):
        col = 0
        while (col < 11944 and graph_size != 0):
            if (data[row][col] != 0):
                graph.add_edge(row, col)
                graph_size -= 1
            col += 1
        row += 1

    nx.draw(graph, with_labels=True, font_weight='light')
    plt.show()