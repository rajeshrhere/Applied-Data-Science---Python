# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:18:08 2018

@author: Rajesh Rajendran
"""

import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite

file_choices = 'Employee_Movie_Choices.txt'
file_relationship = 'Employee_Relationships.txt'

cdata = pd.read_csv(file_choices, sep='\t')
rdata = pd.read_csv(file_relationship, sep='\t', header=None)


# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])


# you can use the following function to plot graphs
# make sure to comment it out before submitting to the autograder
def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    #%matplotlib notebook
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(50,30))
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
        
'''
Question 1
Using NetworkX, load in the bipartite graph from Employee_Movie_Choices.txt and return that graph.

This function should return a networkx graph with 19 nodes and 24 edges
'''
def answer_one():
    gr = nx.Graph()
    gr.add_nodes_from(employees, bipartite=0, type = 'employee')
    gr.add_nodes_from(movies, bipartite=1, type = 'movie')
    
    edges = []
    for i in range(len(cdata)):
        edges.append((cdata.iloc[i][0], cdata.iloc[i][1]))
    
    gr.add_edges_from(edges)
    return gr

'''
Question 2
Using the graph from the previous question, add nodes attributes named 'type' where movies have the value 'movie' and employees have the value 'employee' and return that graph.

This function should return a networkx graph with node attributes {'type': 'movie'} or {'type': 'employee'}
'''
def answer_two():
    gr = answer_one()
    
    return gr

'''
Question 3
Find a weighted projection of the graph from answer_two which tells us how many movies different pairs of employees have in common.

This function should return a weighted projected graph.
''''
def answer_three():
    egr = bipartite.weighted_projected_graph(answer_one(), employees)
    return egr

answer_three().edges(data=True)

'''
Question 4
Suppose you'd like to find out if people that have a high relationship score also like the same types of movies.

Find the Pearson correlation ( using DataFrame.corr() ) between employee relationship scores and the number of movies they have in common. If two employees have no movies in common it should be treated as a 0, not a missing value, and should be included in the correlation calculation.

This function should return a float.
'''
def answer_four():
    df = nx.to_pandas_dataframe(answer_three())
    
    dfl = []
    for i in range(len(rdata)):
        dfl.append(df.loc[rdata.iloc[i][0]][rdata.iloc[i][1]])
        
    rdata[3] = dfl
    
    return (rdata.corr()).iloc[0][3]
    