
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[132]:


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


# In[3]:


def fit_plot(l, func_2_fit, size_ini, size_fin, step):
    l_func_values =[i*func_2_fit(i) for i in range(size_ini, size_fin+1, step)]
    
    lr_m = LinearRegression()
    X = np.array(l_func_values).reshape( len(l_func_values), -1 )
    lr_m.fit(X, l)
    y_pred = lr_m.predict(X)
    
    plt.plot(l, '*', y_pred, '-')

def n2_log_n(n):
    return n**2. * np.log(n)


# In[96]:


l = [
[0, 10, 1, np.inf],
[np.inf, 0, 1, np.inf],
[np.inf, np.inf, 0, 1 ],
[np.inf, 1, np.inf, 0]
]

m_g = np.array(l)

def print_m_g(m_g):
    print("graph_from_matrix:\n")
    n_v = m_g.shape[0]
    for u in range(n_v):
        for v in range(n_v):
            if v != u and m_g[u, v] != np.inf:
                print("(", u, v, ")", m_g[u, v])

d_g = {
0: {1: 10, 2: 1}, 
1: {2: 1}, 
2: {3: 1},
3: {1: 1}
}

def print_d_g(d_g):
    print("\ngraph_from_dict:\n")
    for u in d_g.keys():
        for v in d_g[u].keys():
            print("(", u, v, ")", d_g[u][v])

print_m_g(m_g)
print_d_g(d_g)


# In[151]:



def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):
    """
    """
    grafo_completo = np.around(max_weight * np.random.rand(n_nodes, n_nodes),decimals)
    ramas = np.random.binomial(1, sparse_factor, size=(n_nodes, n_nodes)).astype(np.float32)  # dicta si hay ramas o no

    for u in range(n_nodes):
        for v in range(n_nodes):
            if ramas[u][v] == 1:
                grafo_completo[u][v] = np.inf

    np.fill_diagonal(grafo_completo, 0)

    return grafo_completo

def cuenta_ramas(m_g):
    return len(np.flatnonzero(np.where(m_g == np.inf,0,1))) - m_g.shape[0]


def check_sparse_factor(n_grafos,n_nodes, sparse_factor):
    grafos = [None]*n_grafos
    ramas = [None]*n_grafos
    factores = [None]*n_grafos

    for i in range(n_grafos):
        grafos[i] = rand_matr_pos_graph(n_nodes,sparse_factor,10.,decimals=2)
        ramas[i] = cuenta_ramas(grafos[i])
        factores[i] = ramas[i] / (n_nodes**2 - n_nodes)

    return 1 - np.average(factores)


def m_g_2_d_g(m_g):
    """
    """
    d_g = {}

    n_v = m_g.shape[0]
    for i in range(n_v):
        for j in range(n_v):
            if d_g.get(i) == None:
                d_g.update({i:{}})
            if i != j and m_g[i][j] != np.inf:
                d_g[i].update({j:m_g[i][j]})

    return d_g


def d_g_2_m_g(d_g):
    """
    """
    nkeys = len(d_g.keys())

    m_g = np.empty((nkeys, nkeys), np.float32)

    for i in range(nkeys):
        for j in range(nkeys):
            if d_g[i].get(j) != None and i != j:
                m_g[i][j] = d_g[i].get(j)
            elif i != j:
                m_g[i][j] = np.inf
            else:
                m_g[i][j] = 0

    return m_g


def save_object(obj, f_name="obj.pklz", save_path='.'):
    """"""
    file_path = save_path + f_name

    final_file = gzip.open(file_path, 'wb')

    pickle.dump(obj, final_file)

    final_file.close()



def read_object(f_name, save_path='.'):
    """"""
    file_path = save_path + f_name

    final_file = gzip.open(file_path, 'rb')

    data = pickle.load(final_file)

    final_file.close()

    return data


def d_g_2_TGF(d_g, f_name):
    """
    """
    nNodos = len(d_g.keys())

    data = ''

    for key, value in d_g.items():
        data = data + str(key) + '\n'

    data = data + '#\n'

    for key, value in d_g.items():
        nkeys = len(value.keys())
        for key2, value2 in value.items():
            data = data + str(key) + ' ' + str(key2) + ' ' + str(value2) + '\n'

    save_object(data, f_name)
    

def TGF_2_d_g(f_name):
    """   
    
    """
    
    d_g = {}

    
    data = read_object(f_name)
    
    data = data.split('\n')
    aux  = []
    
    for i in data:
        aux.append(i)
        if i != '#':
            d_g.update({i:{}})
            
        else:
            break
        
    data = data[len(aux):-1]
    
    for i in data:
        s = i.split(' ')
        d_g[s[0]].update({s[1]:s[2]})
    
    return d_g
    

def dijkstra_d(d_g, u):
    """
    """
    d_dist = {}
    d_prev = {}
    distancias = np.full(len(d_g.keys()), np.inf)    
    vistos = np.full(len(d_g.keys()), False)
    padre = np.full(len(d_g.keys()), None)
    
    q = qe.PriorityQueue()
    distancias[u] = 0.
    q.put((0.0,u))
    
    while not q.empty():
        n = q.get()
        vistos[n[1]] = True
        
        for keys,values in d_g[n[1]].items():
            if distancias[keys] > distancias[n[1]] + values:
                distancias[keys] = distancias[n[1]] + values
                padre[keys] = n[1]
                q.put((distancias[keys], keys))
    
    for i in range(len(distancias)):
        d_dist.update({i:distancias[i]})
        d_prev.update({i:padre[i]})
    
    return d_dist,d_prev

def dijkstra_m(m_g,u):
    """
    """
    d_dist = {}
    d_prev = {}
    n_v = m_g.shape[0]
    distancias = np.full(n_v, np.inf)    
    vistos = np.full(n_v, False)
    padre = np.full(n_v, None)
    
    q = qe.PriorityQueue()
    distancias[u] = 0.
    q.put((0.0,u))
    
    while not q.empty():
        n = q.get()
        vistos[n[1]] = True

        for i in m_g[n[1]]:
            if distancias[m_g[n[1]].tolist().index(i)] > distancias[n[1]] + i:
                distancias[m_g[n[1]].tolist().index(i)] = distancias[n[1]] + i
                padre[m_g[n[1]].tolist().index(i)] = n[1]
                q.put((distancias[m_g[n[1]].tolist().index(i)], m_g[n[1]].tolist().index(i)))


    for i in range(len(distancias)):
        d_dist.update({i:distancias[i]})
        d_prev.update({i:padre[i]})
    
    return d_dist,d_prev

def min_paths(d_prev):
    """
    
    d_path = {}
    
    for keys,values in d_prev.items():
        n = keys
        while d_prev[n] != None:
            p.update({keys:p[keys].append(d_prev[n])})
            n = d_prev[n]
    """        
    pass

def time_dijktra_m(n_graphs,n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    grafos = []
    dijktras = []
    i = 0
    n_nodes_act = n_nodes_ini
    
    while n_nodes_act <= n_nodes_fin:
        grafos.append(rand_matr_pos_graph(n_nodes_act, sparse_factor, max_weight=10., decimals=2))
            
        inicio = time.time()
        dijkstra_m(grafos[i],0)
        fin = time.time()
        
        dijktras.append(fin-inicio)
        
        i += 1
        n_nodes_act += step
    
    return dijktras

def time_dijktra_d(n_graphs,n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    grafos = []
    dijktras = []
    i = 0
    n_nodes_act = n_nodes_ini
    
    while n_nodes_act <= n_nodes_fin:
        grafos.append(m_g_2_d_g(rand_matr_pos_graph(n_nodes_act, sparse_factor, max_weight=10., decimals=2)))
            
        inicio = time.time()
        dijkstra_d(grafos[i],0)
        fin = time.time()
        
        dijktras.append(fin-inicio)
        
        i += 1
        n_nodes_act += step
    
    return dijktras


# In[152]:


m_g = rand_matr_pos_graph(n_nodes=5, sparse_factor=0.6, max_weight=10.,decimals = 2)

#print(m_g)
d_g = m_g_2_d_g(m_g)
print("\n\n")
#print(d_g)
# m_g_2 = d_g_2_m_g(d_g)

#print_m_g(m_g)
#print(cuenta_ramas(m_g))
#print(check_sparse_factor(5,5,0.5))
#print(d_g_2_m_g(m_g_2_d_g(m_g)))

#print_d_g(d_g)
#print('\n\n')
# print("\nnum_elem_iguales:\t%d" % (m_g_2 == m_g).sum() )
#save_object(m_g)
#print(read_object('obj.pklz'))
#d_g_2_TGF(d_g, "prueba.pklz")
#TGF_2_d_g("prueba.pklz")

#dist_d ,d_prev = dijkstra_m(m_g,1)
#print(dist_d)
#print('\n\n')
#print(d_prev)

#time_dijktra_m(1000,100, 10000, 10, sparse_factor=.25)

#d = time_dijktra_m(100,1,100,1,sparse_factor=.5)
d = time_dijktra_d(100,1,100,1,sparse_factor=.5)
print(d)

