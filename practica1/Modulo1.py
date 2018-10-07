
# coding: utf-8

# In[1]:


from IPython.display import HTML

HTML('''
<script>code_show=true; 

function code_toggle() {
    if (code_show){
    $('div.input').hide();
    } else {
    $('div.input').show();
    }
    code_show = !code_show
} 

$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>
''')


# # Práctica 1

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


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


# In[4]:


def fit_plot(l, func_2_fit, size_ini, size_fin, step):
    l_func_values =[i*func_2_fit(i) for i in range(size_ini, size_fin+1, step)]
    
    lr_m = LinearRegression()
    X = np.array(l_func_values).reshape( len(l_func_values), -1 )
    lr_m.fit(X, l)
    y_pred = lr_m.predict(X)
    
    plt.plot(l, '*', y_pred, '-')

def n2_log_n(n):
    return n**2. * np.log(n)


# En esta parte vamos a trabajar con grafos ponderados usando dos EdD:
# 
# - Una matriz numpy de adyacencia, donde el elemento i, j indica el peso c_{ij`
#   de la rama (i, j)
# - Un diccionario de diccionarios de diccionarios donde las claves del primer diccionario G son índices de nodos, las claves de los diccionarios G[i] son los vértices de su lista de adyacencia y un diccionario G[i][j] contiene el peso de la rama (i, j).
# 
# Por ejemplo, para el grafo:
# 
# el código
# 
# l = [[0, 10, 1, np.inf],
# [np.inf, 0, 1, np.inf],
# [np.inf, np.inf, 0, 1 ],
# [np.inf, 1, np.inf, 0]]
# 
# m_g = np.array(l)
# 
# generaría su matriz de adyacencia según se describe arriba, mientras que
# 
# d_g = {
# 0: {1: 10, 2:1}, 
# 1: {2: 1}, 
# 2: {3: 1},
# 3: {1: 1}
# }
# 
# generaría su descripción como un dict.

# In[5]:


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


# In[6]:


def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):
    """
    Método que genera grafos de manera aleatoria

    :param n_nodes: Número de nodos que tendrá el grafo
    :param sparse_factor: Proporción de ramas (probabilidad de que una ramas inexistentes)
    :param max_weight: Peso máximo que podría tener una rama
    :param decimals: Número de decimales
    :return: grafo generado
    """
    grafo_completo = np.around(max_weight * np.random.rand(n_nodes, n_nodes),decimals)
    ramas = np.random.binomial(1, sparse_factor, size=(n_nodes, n_nodes)).astype(np.float32)  # dicta si hay ramas o no

    for u in range(n_nodes):
        for v in range(n_nodes):
            if ramas[u][v] == 1:
                grafo_completo[u][v] = np.inf

    np.fill_diagonal(grafo_completo, 0)

    return grafo_completo

def m_g_2_d_g(m_g):
    """
    Método que pasa un grafo en formato de matriz a uno en formato de diccionario

    :param m_g: Grafo en formato de matriz
    :return: Grafo en formato de diccionario
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
    Método que pasa un diccionario en formato de diccionario a uno en formato de matriz

    :param d_g: Grafo en formato de diccionario
    :return: Grafo en formato de matriz
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


################################################# cheking
m_g = rand_matr_pos_graph(n_nodes=5,sparse_factor=0.5,max_weight=50.)
d_g = m_g_2_d_g(m_g)
m_g_2 = d_g_2_m_g(d_g)

print_m_g(m_g)
print_d_g(d_g)
print("\num_elem_iguales:\t%d" % (m_g_2 == m_g).sum() )


# In[7]:


def cuenta_ramas(m_g):
    """
    Método que cuenta las ramas que hay en el grafo

    :param m_g: Grafo
    :return: Número de ramas que tiene el grafo
    """
    return len(np.flatnonzero(np.where(m_g == np.inf,0,1))) - m_g.shape[0]


def check_sparse_factor(n_grafos,n_nodes, sparse_factor):
    """
    Método que calcula la proporción de ramas según una serie de grafos

    :param n_grafos: Número de grafos
    :param n_nodes: Número de nodos de cada grafo
    :param sparse_factor: Proporción de ramas
    :return: La proporción de ramas que hay en los grafos
    """
    grafos = [None]*n_grafos
    ramas = [None]*n_grafos
    factores = [None]*n_grafos

    for i in range(n_grafos):
        grafos[i] = rand_matr_pos_graph(n_nodes,sparse_factor,10.,decimals=2)
        ramas[i] = cuenta_ramas(grafos[i])
        factores[i] = ramas[i] / (n_nodes**2 - n_nodes)

    return 1 - np.average(factores)

############################################################ checking
print(cuenta_ramas(m_g))

n_grafos=50
n_nodes=20
sparse_factor = 0.75

print("\ntrue_sparse_factor: %.3f" % sparse_factor, 
      "\nexp_sparse_factor:  %.3f" % check_sparse_factor(n_grafos=n_grafos, n_nodes=n_nodes, sparse_factor=sparse_factor))


# # 2 Guardando y leyendo grafos
# ## 2.1 Guardando y leyendo grafos con pickle

# In[10]:


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


# ## 2.2 The Trivial Graph Fromat

# In[11]:


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
    
############################################################ checking
f_name = "gr.tgf"
d_g_2_TGF(d_g, f_name)
                       
d_g_2 = TGF_2_d_g(f_name)            
print_d_g(d_g)
print_d_g(d_g_2)


# # 3 Distancias Mínimas en Grafos
# ## 3.1 Programming and Timing Dijstra

# In[16]:


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

############################################################ checking
d_g = {
0: {1: 10, 2: 1}, 
1: {2: 1}, 
2: {3: 1},
3: {1: 1}
}

u_ini = 3

d_dist, d_prev = dijkstra_d(d_g, u_ini)
print(d_dist, '\n', min_paths(d_prev))

d_g_nx = nx.DiGraph()
l_e = [(0, 1, 10), (0, 2, 1), (1, 2, 1), (2, 3, 1), (3, 1, 1)]
d_g_nx.add_weighted_edges_from(l_e)

d, p = nx.single_source_dijkstra(d_g_nx, u_ini, weight='weight')    
print(d, '\n', p)


# ## 3.2 Plotting Dijkstra's Execution Times
# Fit below a linear model An^2 logn+B to the times in the returned lists and plot the real and fitted times discussing the results.

# In[17]:


n_graphs=20
n_nodes_ini=10 
n_nodes_fin=100
step=10
sparse_f= 0.25
l_t_d = time_dijkstra_d(n_graphs=n_graphs, n_nodes_ini=n_nodes_ini, 
                        n_nodes_fin=n_nodes_fin, step=step, sparse_factor=sparse_f)


# In[18]:


fit_plot(l_t_d, n2_log_n, size_ini=n_nodes_ini, size_fin=n_nodes_fin, step=step)


# # 4 The networkx Library
# We are going to use the networkx library to check our Dijkstra results and to get alternative times.
# 
# An example of loading a networkx directed graph is to use a list (i, j, w) of (i, j) edges with weights w can be seen in the following cell:

# In[19]:


g = nx.DiGraph()

l_e = [(0, 1, 10), (0, 2, 1), (1, 2, 1), (2, 3, 1), (3, 1, 1)]
g.add_weighted_edges_from(l_e)

for k1 in g.nodes():
    for k2 in g[k1].keys():
        print('(', k1, k2, ')', g[k1][k2]['weight'])


# In[20]:


def d_g_2_nx_g(d_g):
    l_e = []
    g = nx.DiGraph()
    
    for keys,values in d_g.items():
        for keys2,values2 in values.items():
            l_e.append((keys,keys2,values2))

    g.add_weighted_edges_from(l_e)
    return g
    
def nx_g_2_d_g(nx_g):
    
    d_g = {}
    
    for i in nx_g.nodes():
        for keys,values in nx_g[i].items():
            if d_g.get(i) == None: 
                d_g.update({i:{}})
            for keys2,values2 in values.items():
                d_g[i].update({keys:values2})
                
    return d_g

############################################################ checking
d_g = {
0: {1: 10, 2: 1}, 
1: {2: 1}, 
2: {3: 1},
3: {1: 1}
}

d_g_nx = d_g_2_nx_g(d_g)

print_d_g(d_g)
(d_g_nx)[0][1]


# In[63]:


m_g = rand_matr_pos_graph(n_nodes=4, sparse_factor=0.4, max_weight=10.,decimals = 2)

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
#print(d_prev)

#time_dijktra_m(1000,100, 10000, 10, sparse_factor=.25)

#d = time_dijktra_m(100,1,100,1,sparse_factor=.5)
#d = time_dijktra_d(100,1,100,1,sparse_factor=.5)
#print(d)

print(d_g)
print('\n\n')
g = d_g_2_nx_g(d_g)
print(g[0])
print('\n\n')
print(nx_g_2_d_g(g))

