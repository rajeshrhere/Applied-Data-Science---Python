# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:48:43 2018

@author: Rajesh Rajendran
"""
import pandas as pd
import networkx as nx

file_network = 'email_network.txt'
cdata = pd.read_csv(file_network, sep='\t')
s = []
s = list(cdata['#Sender'].apply(str))
s = s + list(cdata['Recipient'].apply(str))
s = set(s)
edges = []
for i in range(len(cdata)):
    edges.append((str(cdata.iloc[i][0]), str(cdata.iloc[i][1])))

'''
Question 1
Using networkx, load up the directed multigraph from email_network.txt. Make sure the node names are strings.
This function should return a directed multigraph networkx graph.
'''

def answer_one():
    mgr = nx.MultiDiGraph()
    mgr.add_nodes_from(s)
    mgr.add_edges_from(edges)
    
    return mgr

'''
Question 2
How many employees and emails are represented in the graph from Question 1?

This function should return a tuple (#employees, #emails).
'''
def answer_two():
    mgr = answer_one()
    return (len(mgr.nodes()), len(mgr.edges()))

'''
Question 3
Part 1. Assume that information in this company can only be exchanged through email.

When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the receiver, but not vice versa.

Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?

Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways.

Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?

This function should return a tuple of bools (part1, part2).
'''

def answer_three():
    return (nx.is_strongly_connected(answer_one()), nx.is_weakly_connected(answer_one()))
​
answer_three()

'''
Question 4
How many nodes are in the largest (in terms of nodes) weakly connected component?
This function should return an int.
'''
def answer_four():
        
    return len(sorted(nx.weakly_connected_components(answer_one())))   

'''
Question 5
How many nodes are in the largest (in terms of nodes) strongly connected component?

This function should return an int'''
def answer_five():
    df = pd.DataFrame()
    df[0] = sorted(nx.strongly_connected_components(answer_one()))
    df[1] = df[0].apply(len)
    return max(df[1])

'''
Question 6
Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. Call this graph G_sc.

This function should return a networkx MultiDiGraph named G_sc.'''

def answer_six():
        
    G_sc = max(nx.strongly_connected_component_subgraphs(answer_one()), key=len)
    
    return G_sc

'''
Question 7
What is the average distance between nodes in G_sc?

This function should return a float.
'''
def answer_seven():
    return nx.average_shortest_path_length(answer_six())


'''
Question 8
What is the largest possible distance between two employees in G_sc?

This function should return an int.
'''
def answer_eight():
    df = pd.Series(nx.eccentricity(answer_six()))
    df = df.reset_index()
    #df[df[0] == max(df[0])]
    return max(df[0])

answer_eight()

'''
Question 9
What is the set of nodes in G_sc with eccentricity equal to the diameter?

This function should return a set of the node(s).
'''
def answer_nine():
    dia = nx.diameter(answer_six())
    df = pd.Series(nx.eccentricity(answer_six()))
    df = df.reset_index()
    return set(df[df[0] == dia]['index'])

'''
Question 10
What is the set of node(s) in G_sc with eccentricity equal to the radius?

This function should return a set of the node(s).
'''

def answer_ten():
    rad = nx.radius(answer_six())
    df = pd.Series(nx.eccentricity(answer_six()))
    df = df.reset_index()
    return set(df[df[0] == rad]['index'])


g = answer_six()
def get_nodes(nd):
    d = nx.shortest_path_length(g, nd)
    l = []
    for n, i in d.items():
        if i == 1:
            l.append(n)
    return l


'''
Question 11
Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?

How many nodes are connected to this node?

This function should return a tuple (name of node, number of satisfied connected nodes).
'''
def answer_eleven():
    #g = answer_six()
    df = pd.DataFrame()
    df[0] = list(g.nodes())
    df[1] = df[0].apply(get_nodes)
    df[2] = df[1].apply(len)
    df1 = df[df[2] == max(df[2])]

    return (df1.iloc[0][0], df1.iloc[0][1])

'''
Question 12
Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or the center nodes)?

This function should return an integer.

'''
def answer_twelve():
        
    # Your Code Here
    
    return 

'''
Question 13
Construct an undirected graph G_un using G_sc (you can ignore the attributes).

This function should return a networkx Graph.
'''

def answer_thirteen():
        
    # Your Code Here
    
    return nx.Graph(answer_six().to_undirected())
​
answer_thirteen()

'''
Question 14
What is the transitivity and average clustering coefficient of graph G_un?

This function should return a tuple (transitivity, avg clustering).
'''
def answer_fourteen():
        
    g = answer_thirteen()
    
    return (nx.transitivity(g), nx.average_clustering(g))
​
answer_fourteen()



#df = nx.to_pandas_dataframe(answer_one())
#writer = pd.ExcelWriter('df.xlsx')
#df.to_excel(writer, sheet_name='Sheet1')
#writer.save()
