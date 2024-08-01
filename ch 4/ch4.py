# importing some necessary libraries to start
import networkx as nx
import numpy as np
import random
random.seed(0)
np.random.seed(0)

G = nx.erdos_renyi_graph(10, 0.3, seed = 1, directed = False)

# a method to randomly select the next node
# in a graph based on the previous node, and the two parameters
# p and q
def next_node(previous, current, p, q):
    # retrieving the list of neighboring nodes
    # from the current node and initialise the list
    # of alpha values
    neighbors = list(G.neighbors(current))
    alphas = []

    # for each neighbor, calculate appropriate alpha value ie,
    # 1/p -> if neighbor is previous node,
    # 1 -> if neighbor is connected to previous node,
    # 1/q -> otherwise
    for neighbor in neighbors:
        if neighbor == previous:
            alpha = 1/p
        elif G.has_edge(neighbor, previous):
            alpha = 1
        else:
            alpha = 1/q
        
        alphas.append(alpha)

    # now we normalise these values to create probabilities
    probs = [alpha / sum(alphas) for alpha in alphas]

    # now we randomly select the next node based on the transition
    # probabilities calculated in the previous step
    next = np.random.choice(neighbors, size=1, p = probs)[0]

    return next