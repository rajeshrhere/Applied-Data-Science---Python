# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:27:29 2018

@author: Rajesh Rajendran
"""
'''
Assignment 3
In this assignment you will explore measures of centrality on two networks, a friendship network in Part 1, and a blog network in Part 2.

Part 1
Answer questions 1-4 using the network G1, a network of friendships at a university department. Each node corresponds to a person, and an edge indicates friendship.

The network has been loaded as networkx graph object G1.

In [2]:
'''

import pandas as pd
import networkx as nx

G1 = nx.read_gml('friendships.gml')

'''
Question 1
Find the degree centrality, closeness centrality, and normalized betweeness centrality (excluding endpoints) of node 100.

This function should return a tuple of floats (degree_centrality, closeness_centrality, betweenness_centrality).

In [4]:
'''

def answer_one():
    return (nx.degree_centrality(G1)[100], nx.closeness_centrality(G1)[100],nx.betweenness_centrality(G1)[100])

answer_one()

'''
Question 2
Suppose you are employed by an online shopping website and are tasked with selecting one user in network G1 to send an online shopping voucher to. We expect that the user who receives the voucher will send it to their friends in the network. You want the voucher to reach as many nodes as possible. The voucher can be forwarded to multiple users at the same time, but the travel distance of the voucher is limited to one step, which means if the voucher travels more than one step in this network, it is no longer valid. Apply your knowledge in network centrality to select the best candidate for the voucher.

This function should return an integer, the name of the node.
'''
def answer_two():
    dc = nx.degree_centrality(G1)
    df = pd.DataFrame()
    df[0] = sorted(dc)
    dcf = lambda x : dc[x]
    df[1] = df[0].apply(dcf)
    
    df = df.sort_values([1], ascending=False)
    df = df.head(1)
    
    return df.iloc[0][0]

answer_two()

'''
Question 3
Now the limit of the voucher’s travel distance has been removed. Because the network is connected, regardless of who you pick, every node in the network will eventually receive the voucher. However, we now want to ensure that the voucher reaches the nodes in the lowest average number of hops.

How would you change your selection strategy? Write a function to tell us who is the best candidate in the network under this condition.

This function should return an integer, the name of the node.
'''
def answer_three():
    dc = nx.closeness_centrality(G1)
    df = pd.DataFrame()
    df[0] = sorted(dc)
    dcf = lambda x : dc[x]
    df[1] = df[0].apply(dcf)
    
    df = df.sort_values([1], ascending=False)
    df = df.head(1)
    
    return df.iloc[0][0]

answer_three()
'''
Question 4
Assume the restriction on the voucher’s travel distance is still removed, but now a competitor has developed a strategy to remove a person from the network in order to disrupt the distribution of your company’s voucher. Your competitor is specifically targeting people who are often bridges of information flow between other pairs of people. Identify the single riskiest person to be removed under your competitor’s strategy?

This function should return an integer, the name of the node.
'''''''''''''''

def answer_four():
    dc = nx.betweenness_centrality(G1)
    df = pd.DataFrame()
    df[0] = sorted(dc)
    dcf = lambda x : dc[x]
    df[1] = df[0].apply(dcf)
    
    df = df.sort_values([1], ascending=False)
    df = df.head(1)
    
    return df.iloc[0][0]

answer_four()


'''
Part 2
G2 is a directed network of political blogs, where nodes correspond to a blog and edges correspond to links between blogs. Use your knowledge of PageRank and HITS to answer Questions 5-9.
'''

G2 = nx.read_gml('blogs.gml')

'''
Question 5
Apply the Scaled Page Rank Algorithm to this network. Find the Page Rank of node 'realclearpolitics.com' with damping value 0.85.

This function should return a float.
'''

def answer_five():
        
    return nx.pagerank(G2)['realclearpolitics.com']

answer_five()

'''
Question 6
Apply the Scaled Page Rank Algorithm to this network with damping value 0.85. Find the 5 nodes with highest Page Rank.

This function should return a list of the top 5 blogs in desending order of Page Rank.
'''
def answer_six():
    G = nx.pagerank(G2)
    pr = lambda x : G[x]
    df = pd.DataFrame()
    df[0] = sorted(nx.pagerank(G2))
    df[1] = df[0].apply(pr)
    df = df.sort_values([1], ascending=False)
    
    df = df.head()
    return list(df[0])

answer_six()
'''
Question 7
Apply the HITS Algorithm to the network to find the hub and authority scores of node 'realclearpolitics.com'.

Your result should return a tuple of floats (hub_score, authority_score).
'''
def answer_seven():
    h = nx.hits(G2)
    hub = h[0]
    auth = h[1]
    
    return (hub['realclearpolitics.com'], auth['realclearpolitics.com'])

answer_seven()

'''
Question 8
Apply the HITS Algorithm to this network to find the 5 nodes with highest hub scores.

This function should return a list of the top 5 blogs in desending order of hub scores.
'''
def answer_eight():
    G = nx.hits(G2)[0]
    
    pr = lambda x : G[x]
    df = pd.DataFrame()
    df[0] = sorted(G)
    df[1] = df[0].apply(pr)
    df = df.sort_values([1], ascending=False)
    
    df = df.head()
    
    return list(df[0])# Your Answer Here

answer_eight()

'''
Question 9
Apply the HITS Algorithm to this network to find the 5 nodes with highest authority scores.

This function should return a list of the top 5 blogs in desending order of authority scores.
'''
def answer_nine():
    G = nx.hits(G2)[1]
    
    pr = lambda x : G[x]
    df = pd.DataFrame()
    df[0] = sorted(G)
    df[1] = df[0].apply(pr)
    df = df.sort_values([1], ascending=False)
    
    df = df.head()
    
    return list(df[0])# Your Answer Here

answer_nine()

