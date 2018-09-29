import matplotlib as matplotlib

#matplotlib inline
#load_ext autoreload
#autoreload 2

import string, random
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pickle
import gzip

import queue as qe

from sklearn.linear_model import LinearRegression

import networkx as nx


def fit_plot(l, func_2_fit, size_ini, size_fin, step):
    l_func_values = [i * func_2_fit(i) for i in range(size_ini, size_fin + 1, step)]

    lr_m = LinearRegression()
    X = np.array(l_func_values).reshape(len(l_func_values), -1)
    lr_m.fit(X, l)
    y_pred = lr_m.predict(X)

    plt.plot(l, '*', y_pred, '-')


def n2_log_n(n):
    return n ** 2. * np.log(n)

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


#print_m_g(m_g)
#print_d_g(d_g)

def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):
    """
    """
    grafo_completo = np.random.random_integers(1, max_weight, (n_nodes, n_nodes))
    decimales = np.around(np.random.random_sample((n_nodes,n_nodes)),decimals)
    ramas = np.random.binomial(1, sparse_factor, size=(n_nodes, n_nodes))  # dicta si hay ramas o no

    grafo_completo = np.add(grafo_completo,decimales)

    for u in range(n_nodes):
        for v in range(n_nodes):
            if ramas[u][v] == 1:
                grafo_completo[u][v] = np.inf

    np.fill_diagonal(grafo_completo, 0)
    #print(grafo_completo)
    #print("\n")
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
            if i != j and m_g[i][j] != np.inf:
               if d_g.get(i) == None:
                   d_g.update({i:{j:m_g[i][j]}})
               else:
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



m_g = rand_matr_pos_graph(n_nodes=5, sparse_factor=0.5, max_weight=10.,decimals = 2)

print(m_g)
d_g = m_g_2_d_g(m_g)
# m_g_2 = d_g_2_m_g(d_g)

#print_m_g(m_g)
#print(cuenta_ramas(m_g))
#print(check_sparse_factor(5,5,0.5))
#print(d_g_2_m_g(m_g_2_d_g(m_g)))
#mu1 = np.random.randint(0, 5, (5, 5))
#mu2 = np.random.binomial(1, 0.5, size=(5, 5))
#print(mu1)
#print(mu2)
print('\n\n')
print(d_g)
print('\n\n')
# print("\nnum_elem_iguales:\t%d" % (m_g_2 == m_g).sum() )
#save_object(m_g)
#print(read_object('obj.pklz'))
d_g_2_TGF(d_g)


