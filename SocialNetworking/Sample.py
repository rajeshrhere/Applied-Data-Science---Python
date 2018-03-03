


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:28:20 2018

@author: Rajesh Rajendran
"""

import networkx as nx
from networkx.algorithms import bipartite
B = nx.Graph()
B.add_nodes_from(['A', 'B', 'C', 'D', 'E'], bipartite=0)
B.add_nodes_from(['1', '2', '3', '4'], bipartite=1)
B.add_edges_from([('A', 1), ('B', 1), ('C', 1), ('C', 3), ('D', 2), ('E', 3), ('E', 4)])

#bipartite.sets(B)

def plot_graph(G, weight_name=None, size=(15, 10)):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    #%matplotlib notebook
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(size[0],size[1]))
    #plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nx.draw_networkx(G, pos, edges=edges);
        

x = set(['A', 'B', 'C', 'D', 'E'])

P = bipartite.projected_graph(B, x)
P1 = bipartite.weighted_projected_graph(B, x)

g = nx.Graph()
g.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
g.add_edges_from([('A', 'E'),
('A', 'D'),
('A', 'C'),
('B', 'D'),
('C', 'G'),
('D', 'E'),
('D', 'G'),
('D', 'H'),
('E', 'H'),
('F', 'H')])

g.node['A']['community'] = 0 
g.node['B']['community'] = 0
g.node['C']['community'] = 0
g.node['D']['community'] = 0
g.node['G']['community'] = 0
g.node['E']['community'] = 1
g.node['F']['community'] = 1
g.node['H']['community'] = 1
list(nx.ra_index_soundarajan_hopcroft(g))
list(nx.cn_soundarajan_hopcroft(g))
g.degree()    

g = nx.DiGraph()
g.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
g.add_edges_from([('A', 'B'),
('A', 'D'),
('A', 'C'),
('B', 'A'),
('B', 'C'),
('C', 'A'),
('D', 'E'),
('E', 'C')])
g.degree()

g = nx.DiGraph()
g.add_nodes_from(['A', 'B', 'C', 'D'])
g.add_edges_from([( 'A', 'B'),( 'A', 'C'),( 'C', 'A'),( 'B', 'C'),( 'D', 'C')])






def GetEdges(g, e):
    edgelist = list(g.edges())
    vlist = []
    for i in range(0, len(edgelist)):
        if edgelist[i][0] == e or edgelist[i][1] == e :
            vlist.append(edgelist[i])
    return vlist


def MakeGraph(vlist):
    g = nx.Graph()
    df = pd.DataFrame(vlist)
    g.add_nodes_from(set(df[0].append(df[1])))
    g.add_edges_from(vlist)
    return g

    


#Creating a feature matrix from a networkx graph
#In this notebook we will look at a few ways to quickly create a feature matrix from a networkx graph.
import networkx as nx
import pandas as pd
​
G = nx.read_gpickle('major_us_cities')
#Node based features

G.nodes(data=True)

# Initialize the dataframe, using the nodes as the index
df = pd.DataFrame(index=G.nodes())
#Extracting attributes
#Using nx.get_node_attributes it's easy to extract the node attributes in the graph into DataFrame columns.

df['location'] = pd.Series(nx.get_node_attributes(G, 'location'))
df['population'] = pd.Series(nx.get_node_attributes(G, 'population'))
​
df.head()


df['clustering'] = pd.Series(nx.clustering(G))
df['degree'] = pd.Series(G.degree())
df

#Edge based features

G.edges(data=True)

# Initialize the dataframe, using the edges as the index
df = pd.DataFrame(index=G.edges())
#Extracting attributes
#Using nx.get_edge_attributes, it's easy to extract the edge attributes in the graph into DataFrame columns.


df['weight'] = pd.Series(nx.get_edge_attributes(G, 'weight'))
​
df

#Creating edge based features
#Many of the networkx functions related to edges return a nested data structures. We can extract the relevant data using list comprehension.


df['preferential attachment'] = [i[2] for i in nx.preferential_attachment(G, df.index)]
​
df

#In the case where the function expects two nodes to be passed in, we can map the index to a lamda function.

df['Common Neighbors'] = df.index.map(lambda city: len(list(nx.common_neighbors(G, city[0], city[1]))))
​
df
