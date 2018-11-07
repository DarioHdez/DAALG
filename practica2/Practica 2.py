
# coding: utf-8

# In[1]:

import string, random
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import gzip
import pickle
import itertools
import queue as qe

from sklearn.linear_model import LinearRegression

import networkx as nx


# In[2]:

mg = nx.MultiDiGraph()

mg.add_weighted_edges_from([(0, 1, 10), (0, 2, 1), (1, 2, 1), (2, 3, 1), (3, 1, 1)])
mg.add_weighted_edges_from([(0, 0, 10), (0, 2, 1), (1, 2, 1), (2, 3, 1), (3, 1, 1)])

print(mg[0])


# In[3]:

def m_mg_2_d_mg(m_mg):
    ret = {}
    
    for i in range(m_mg.shape[0]):
        ret.update({i:{}})
        for j in m_mg[i]:
            if j > 0:
                ret[i].update({j:{}})
                for k in range(j):
                    ret[i][j].update({k:1})
                    
    return ret
            
def rand_unweighted_multigraph(n_nodes, num_max_multiple_edges = 3, prob = 0.5):
    
    ramas = np.random.binomial(num_max_multiple_edges-1, 1-prob, size=(n_nodes, n_nodes)) 

    return m_mg_2_d_mg(ramas)


# In[123]:

g = rand_unweighted_multigraph(4)


# In[44]:

def graph_2_multigraph(g):
    ret = {}
    
    for u in g.keys():
        ret.update({u:{}})
        for v,i in g[u].items():
            ret[u].update({v:{0:i}})
    
    return ret

def print_multi_graph(g):
    for u in g.keys():
        for v in g[u].keys():
            l = []
            for i in g[u][v].keys():
                l.append(str(i)+':'+str(g[u][v][i]))
            print((u,v),"".join(str(x)+" " for x in l))


# In[124]:

'''d_g = {
0: {1: 10, 2: 1}, 
1: {2: 1, 3: 5}, 
2: {3: 1},
3: {1: 1, 0:20}
}'''

#print(graph_2_multigraph(d_g)
#mg = graph_2_multigraph(d_g)
print(g)
print('\n\n')

print_multi_graph(g)


# In[125]:

def adj_inc_directed_multigraph(d_mg):
    
    adj = [0]*len(d_mg)
    inc = [0]*len(d_mg)
    
    for key_n,value_n in d_mg.items():
        for key_adj,value_adj in value_n.items():
            for keys_edg in value_adj.keys():
                adj[key_n]+=1
                inc[key_adj]+=1
    
    return adj,inc

def isthere_euler_path_directed_multigraph(d_mg):
    odd = 0
    adj, inc = adj_inc_directed_multigraph(d_mg)
    for adj_n, inc_n in zip(adj, inc):
        result = adj_n + inc_n
        if (result % 2) != 0:
            odd += 1
    if odd != 2:
        return False
    else:
        return True

    
def first_last_euler_path_directed_multigraph(d_mg):
    ret = []
    i = 0
    adj, inc = adj_inc_directed_multigraph(d_mg)
    if isthere_euler_path_directed_multigraph(d_mg):
        for adj_n, inc_n in zip(adj, inc):
            result = adj_n + inc_n
            if (result % 2) != 0:
                ret.append(i)
            i += 1
        return tuple(ret)
    else:
        return ()
                
    
adj, inc = adj_inc_directed_multigraph(g);
print(adj,"\n", inc)
print(isthere_euler_path_directed_multigraph(g))
print(first_last_euler_path_directed_multigraph(g))
    


# In[ ]:

def ewdm_rec(used_paths,u,d_mg):
    pass

def euler_walk_directed_multigraph(u, d_mg):
    
    used_paths = []
    
    if u not in d_mg.keys(): return []
    
    for keys,values in d_mg[u].items():
        for keys_n, values_n in values.items():
            

