#!/usr/bin/env python
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
import copy

from sklearn.linear_model import LinearRegression
from random import shuffle

import networkx as nx


# In[2]:


mg = nx.MultiDiGraph()

mg.add_weighted_edges_from([(0, 1, 10), (0, 2, 1), (1, 2, 1), (2, 3, 1), (3, 1, 1)])
mg.add_weighted_edges_from([(0, 0, 10), (0, 2, 1), (1, 2, 1), (2, 3, 1), (3, 1, 1)])

print(mg[0])


# In[3]:


def m_mg_2_d_mg(m_mg):
    """
    Funcion que pasa de una matriz que representa un multigrafo dirigido a un diccionario.
     En el diccionario de retorno, los datos se guardan de la siguiente manera:
     El primer diccionario guarda los nodos, el segundo diccionario guarda los nodos a los que se accede
      desde cada uno de los nodos, y en el ultimo diccionario se guardan las ramas.
    
    :param m_mg matriz a transformar
    :return diccionario de diccionario de diccionario con la representacion de los datos del grafo
    """
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
    """
    Funcion que genera un multigrafo dirigido de manera aleatoria. 
    El multigrafo se crea como una matriz, y se llama a la funcion m_mg_2_d_mg para
     transformarlo en un diccionario.
    
    :param n_nodes numero de nodos que tendrá el grafo
    :param num_max_multiple_edges numero maximo de ramas que tendra cada nodo, incluyendo adyacencias e incidencias
    :param prob probabilidad de que se genere una rama
    :return diccionario con el multigrafo dirigido
    """
    ramas = np.random.binomial(num_max_multiple_edges-1, 1-prob, size=(n_nodes, n_nodes)) 

    return m_mg_2_d_mg(ramas)


# In[4]:


g = rand_unweighted_multigraph(4)


# In[5]:


def graph_2_multigraph(g):
    """
    Funcion que transforma un grafo en un multigrafo añadiéndole un diccionario adicional en el que
     guardar las ramas
    
    :param g grafo a transformar
    :retorno diccionario con el multigrafo dirigido
    """
    ret = {}
    
    for u in g.keys():
        ret.update({u:{}})
        for v,i in g[u].items():
            ret[u].update({v:{0:i}})
    
    return ret

def print_multi_graph(g):
    """
    Funcion para imprimir por pantalla con formato mas legible un multigrafo.
     El formato de impresion es:
      (inicio,final) nºRama : Peso 
    
    :param g multigrafo dirigido a imprimir
    """
    for u in g.keys():
        for v in g[u].keys():
            l = []
            for i in g[u][v].keys():
                l.append(str(i)+':'+str(g[u][v][i]))
            print((u,v),"".join(str(x)+" " for x in l))


# In[6]:


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


# In[7]:


def adj_inc_directed_multigraph(d_mg):
    """
    Funcion que devuelve las listas de adyacencia e incidencia de un multigrafo dirigido
     Cada lista tiene como indice el nodo que representa del multigrafo, y para cada indice tiene los valores
     de adyacencia e incidencia de ese nodo
     
    :param d_mg multigrafo dirigido del que sacar las listas
    :return lista con las adyacencias , lista con las incidencias
    """
    adj = [0]*len(d_mg)
    inc = [0]*len(d_mg)

    
    for key_n,value_n in d_mg.items():
        for key_adj,value_adj in value_n.items():
            for keys_edg in value_adj.keys():
                adj[key_n]+=1
                inc[key_adj]+=1
    
    return adj,inc

def _get_u_v(inc,adj):
    """
    Funcion privada que devuelve el nodo inicial y final en un camino euleriano para un multigrafo dirigido en base a sus
     listas de incidenia y adyacencia
    
    :param inc lista de incidencias
    :param adj lista de adyacencias
    :return nodo inical,nodo final ambos representados como enteros
    """
    u = None
    v = None # índice del último nodo
     
    for i in range(len(adj)):
        # check for u
        if inc[i] == adj[i] - 1 and u == None:
            u = i
            
        # check for v
        if inc[i] == adj[i] + 1 and v == None:
            v = i
    
    return u,v

def isthere_euler_path_directed_multigraph(d_mg):
    """
    Funcion que comprueba si un multigrafo dirigido tiene un camino euleriano.
     Primero encuentra la lista de adyacencia e incidencia del multigrafo, despues el nodo inicial y final,
     y luego comprueba que el resto de nodos tengan el mismo numero de incidencias que de adyacencias
     
    :param d_mg multigrafo dirigido al que comprobar 
    :return True si tiene camino, False en caso contrario
    """
    adj, inc = adj_inc_directed_multigraph(d_mg)
   
    u,v = _get_u_v(inc,adj)
    
    if u == None or v == None: # No tenemos nodo inicial/final
        return False
    
    return all([True if adj[n] == inc[n] or n == u or n == v else False for n in range(len(adj))])
    
    
def first_last_euler_path_directed_multigraph(d_mg):
    """
    Funcion que te devuelve el nodo inicial y el nodo final de un multigrafo en caso de que tenga un camino euleriano.
    
    :param d_mg multigrafo dirigido del que encontrar el nodo inicial y final
    :return nodo inicial y final en formato tupla (inicial,final), tupla vacia en caso de no tener camino euleriano
    """
    if not isthere_euler_path_directed_multigraph(d_mg): 
        return ()
    else:
        adj, inc = adj_inc_directed_multigraph(d_mg)
        u,v = _get_u_v(inc,adj)
        
        return (u,v)

    
d_g_camino = { # sabemos que este grafo tiene camino euleriano
0: {1:{0:1}, 3:{0:1}}, 
1: {1:{0:1,1:1}, 2:{0:1}}, 
2: {0:{0:1}},
3: {}
}
    
adj, inc = adj_inc_directed_multigraph(g);
print(adj,"\n", inc)
print(isthere_euler_path_directed_multigraph(g))
adj, inc = adj_inc_directed_multigraph(d_g_camino);
print(adj,"\n", inc)
print(isthere_euler_path_directed_multigraph(d_g_camino))
print(first_last_euler_path_directed_multigraph(g))
print(first_last_euler_path_directed_multigraph(d_g_camino))
    


# In[8]:


d_g_camino = { # sabemos que este grafo tiene camino euleriano
0: {1:{0:1}, 3:{0:1}}, 
1: {1:{0:1,1:1}, 2:{0:1}}, 
2: {0:{0:1}},
3: {}
}

d_g_camino = { # sabemos que este grafo tiene camino euleriano
0: {1:{0:1}}, 
1: {2:{0:1}, 3:{0:1}}, 
2: {3:{0:1}},
3: {5:{0:1}, 6:{0:1}},
4: {1:{0:1}},
5: {},
6: {4:{0:1}}
}

def euler_walk_directed_multigraph(u, d_mg):
    """
    Funcion que "anda" por el grafo apuntando las ramas por las que pasa a partir de un nodo dado
    
    :param u nodo inicial desde el que andar
    :param d_mg multigrafo dirigido del que andar
    :return lista con las ramas que ha utilzado, en formato (inicial,final) para cada rama
    """
    if not d_mg or u == None:
        return []
    
    c = copy.deepcopy(d_mg)
    ret =  []
    
    while d_mg[u].keys():
        if not list(c[u].keys()):
            break
        
        z = list(c[u].keys())[0]
        p = list(c[u][z])[0]
        
        ret.append((u,z))
        c[u][z].pop(p)
        
        if not c[u][z].keys(): 
            c[u].pop(z)
        
        u = z
        
    return ret

def _remove_path_from_multigraph(l_path,d_mg):
    """
    Funcion que elimina un camino de un multigrafo dirigido. El multigrafo quedara solo con los nodos y diccionarios 
    vacios si se eliminan todas las ramas posibles.
    
    :param l_path camino a quitar del multigrafo
    :param d_mg multigrafo dirigido del que quitar el camino
    :return esta funcion no devuelve nada
    """
    for tup in l_path:
        if d_mg[tup[0]]:
            if tup[1] in d_mg[tup[0]]:
                if len(d_mg[tup[0]][tup[1]]) == 1:
                    d_mg[tup[0]].pop(tup[1])
                else:
                    extract_key = list(d_mg[tup[0]][tup[1]].keys())[0]
                    # Extraemos la key
                    d_mg[tup[0]][tup[1]].pop(extract_key)


def next_first_node(l_path,d_mg):
    """
    Funcion que devuelve el siguiente nodo del que andar en un multigrafo dirigido, si antes quitas todas
     las ramas presentes en una lista dada
     
    :param l_path lista con las ramas visitadas previamente
    :param d_mg multigrafo dirigido 
    :return entero con el siguiente nodo accesible del que andar, None si no hay ninguno 
    """
    if not l_path or not d_mg:
        return None
    
    _remove_path_from_multigraph(l_path,d_mg)

    if first_last_euler_path_directed_multigraph(d_mg):

        (first,last) = first_last_euler_path_directed_multigraph(d_mg)
        return first


    for k, v in d_mg.items():
        if v.keys():
            return k
    
    return None

def path_stitch(path_1, path_2):
    """
    Funcion que junta 2 listas de ramas visitadas. Encuentra el punto en el que el segundo camino coincide al principio
     y al final dentro del primer camino, como si fuese un circuito interno en el grafo.
    
    :param path_1 primera lista con el camino
    :param path_2 segunda lista con el camino a insertar
    :return lista con los dos caminos combinados de manera correspondiente, una lista vacia si los 2 argumentos son pasados de manera incorrecta
    """
    if not path_1:
        return path_2
    if not path_2:
        return path_1
    if (not path_1) and (not path_2):
        return []
    
    ret = []
    rep = 1
    
    if path_1[0][0] == path_2[-1][1]: 
        ret = path_2
        for tup in path_1:
            ret.append(tup)
        
        return ret

    for tup1 in path_1:
        if tup1[0] == path_2[0][1] and rep and len(path_2) == 1:
            ret.append(path_2[0])
            rep = 0
            
        ret.append(tup1)
        if tup1[1] == path_2[0][0] and rep:
            for tup2 in path_2:
                ret.append(tup2)
                rep = 0
            
    while path_1 == ret:
        path_2.append(path_2.pop(0))
#         print('\nPath2 queda: ',path_2)
        ret.clear()
        
        if path_1[0][0] == path_2[-1][1]: 
            ret = path_2
            for tup in path_1:
                ret.append(tup)

            return ret
        
        for tup1 in path_1:
            ret.append(tup1)
            if tup1[1] == path_2[0][0]:
                for tup2 in path_2:
                    ret.append(tup2)
#         print('\nRet after for: ',ret)
        
    return ret

print(d_g_camino)
#print(euler_walk_directed_multigraph(0,d_g_camino))
print(euler_walk_directed_multigraph(1,d_g_camino))
#print(euler_walk_directed_multigraph(3,d_g_camino))
#print(next_first_node(euler_walk_directed_multigraph(0,d_g_camino), d_g_camino))
l1 = euler_walk_directed_multigraph(1,d_g_camino)
n = next_first_node(l1, d_g_camino)

# print(euler_walk_directed_multigraph(n,d_g_camino))
l2 = euler_walk_directed_multigraph(n,d_g_camino)
print(l2)
print(path_stitch(l1, l2))
print(next_first_node(path_stitch(l1, l2),d_g_camino))

# l1 = [(10, 7), (7, 7), (7, 7), (7, 7), (7, 11), (11, 3), (3, 9), (9, 3), (3, 9), (9, 3), (3, 12), (12, 4), (4, 5), (5, 0), (0, 1), (1, 11), (11, 14), (14, 6), (6, 0), (0, 1), (1, 8), (8, 4), (4, 7), (7, 11), (11, 10), (10, 7)]
# l2 = [(2,3)]

# print(path_stitch(l1,l2))


# In[9]:


d_g_camino = { # sabemos que este grafo tiene camino euleriano
0: {1:{0:1}},
1: {2:{0:1}}, 
2: {3:{0:1}, 4:{0:1}}, 
3: {4:{0:1}},
4: {6:{0:1}, 7:{0:1}},
5: {2:{0:1}},
6: {},
7: {5:{0:1}}
}

def euler_path_directed_multigraph(d_mg):
    # Comprobamos que tenga camino euleriano
    if not isthere_euler_path_directed_multigraph(d_mg):
        print("NO ES CAMINO EULERIANO")
        return []
    
    # Sacamos el nodo inicial y el nodo final 
    (first,last) = first_last_euler_path_directed_multigraph(d_mg)
#     print('\nFirst: ',first,'Last: ', last)
    
    l1 = euler_walk_directed_multigraph(first, d_mg)

    n = next_first_node(l1,d_mg)
    
#     print('\nNext Node: ',n)
    
#     print('\nMultigrafo: ',d_mg)
    
    while n != None: # Como mucho lo hace estas veces
        l = euler_walk_directed_multigraph(n,d_mg)
#         print('\nNext walk: ',d_mg)
#         print('\nStitching: ',l1,' with ', l)
        l1 = path_stitch(l1,l)
#         print('\nStiching: ',l1)
        
        n = next_first_node(l,d_mg)        
        
        if n == None:
            return l1
    
    return l1

def __check_euler_path(final_path,d_mg):
    ramas = len(final_path)
    ramas_grafo = 0
    
    for key,value in d_mg.items():
        ramas_grafo += len(value.keys())
    
    print (ramas,ramas_grafo)

# l1 = euler_path_directed_multigraph(d_g_camino)
# # __check_euler_path(l1,d_g_camino)
# print(l1)

d_camino= {
0: {1:{0:1}, 2:{0:1}}, 
1: {}, 
2: {0:{0:1}},
}

d_infinito = {
0: {1: {0: 1, 1: 1}}, 
1: {8: {0: 1}, 11: {0: 1}}, 
2: {3: {0: 1}, 10: {0: 1}}, 
3: {9: {0: 1, 1: 1}, 12: {0: 1, 1: 1}}, 
4: {5: {0: 1}, 7: {0: 1}, 8: {0: 1, 1: 1}}, 
5: {0: {0: 1}}, 
6: {0: {0: 1}, 2: {0: 1, 1: 1}, 13: {0: 1}}, 
7: {7: {0: 1, 1: 1, 2: 1}, 11: {0: 1, 1: 1}}, 
8: {4: {0: 1}, 9: {0: 1, 1: 1}, 12: {0: 1}},
9: {3: {0: 1, 1: 1}, 6: {0: 1, 1: 1}, 14: {0: 1}},
10: {7: {0: 1, 1: 1}, 8: {0: 1}},
11: {3: {0: 1}, 10: {0: 1}, 14: {0: 1}},
12: {4: {0: 1, 1: 1}, 9: {0: 1}, 12: {0: 1, 1: 1, 2: 1}},
13: {4: {0: 1}}, 
14: {6: {0: 1, 1: 1}}
}

print(euler_path_directed_multigraph(d_camino))


# In[10]:


d_g_camino = { # sabemos que este grafo tiene camino euleriano
0: {1:{0:1}},
1: {2:{0:1}}, 
2: {3:{0:1}}, 
3: {4:{0:1}},
4: {5:{0:1}},
5: {6:{0:1}},
6: {7:{0:1}},
7: {0:{0:1}}
}

def isthere_euler_circuit_directed_multigraph(d_mg):
    adj,inc = adj_inc_directed_multigraph(d_mg)
    
    if not adj == inc:
        return False
    else:
        return True

isthere_euler_circuit_directed_multigraph(d_g_camino)

def euler_circuit_directed_multigraph(d_mg, u=0):
    if not isthere_euler_circuit_directed_multigraph(d_mg):
        print("NO ES CIRCUITO")
        return []
    
    l1 = euler_walk_directed_multigraph(u, d_mg)
    n = next_first_node(l1,d_mg)
    
    while n:
        l = euler_walk_directed_multigraph(n,d_mg)
        
        l1 = path_stitch(l1,l)
        
        n = next_first_node(l1,d_mg)
    
    return l1

print(euler_circuit_directed_multigraph(d_g_camino))


# In[11]:


#Cuestion


# In[12]:


def random_sequence(len_seq):
    seq = ['A', 'C', 'G', 'T']
    
    np.random.seed()
    
    return ''.join([seq[np.random.randint(0,4)] for n in range(len_seq)])

s = random_sequence(14)
print('Cadena aleatoria: ' + s)

def spectrum(sequence, len_read):
        
    ret = [sequence[i:i+len_read] for i in range(len(sequence)) if ((i+len_read) <= len(sequence))]
    
    shuffle(ret)
    
    return ret

s = spectrum(s, 3)
# print('\nEspectro X: ', s)

def spectrum_2(spectr):
    sp_list = [spectr[i][j:j+len(spectr[0])-1] for i in  range(len(spectr)) for j in [0,1]]
    sp_menos_1 = []
    
    for i in sp_list:
        if i not in sp_menos_1:
            sp_menos_1.append(i) # Quitamos los duplicados porque un set lo desordena
    
    return sp_menos_1

# print('\nEspectro X-1: ',set(spectrum_2(s)))

def __can_combine(s1,s2):
    if len(s1) == 1: 
        return True
    
    take = len(s1)-1
    #print('can combine',s1,s2,'?')
    #print(s1[-take:],s2[:take])
    #print(s1[-take:] == s2[:take])
    
    return s1[-take:] == s2[:take]

# print(__can_combine('TT','CC'))
# print(__can_combine('AC','GT'))
# print(__can_combine('AG','GC'))

def __combine_spectr(s1,s2):
    if len(s1) == 1:
        return s1+s2
    
    take = len(s1)-1
    #print('lets combine ',s1,s2)
    #print('combine:',s1+s2[take:])
    return s1+s2[take:]

def spectrum_2_graph(spectr):
#     print('\nEspectro X: ', spectr)
    sp_list = spectrum_2(spectr)
                
#     print('\nEspectro X-1: ',sp_list)
    
#     ramas_x_vuelta = 0
    nodos = range(len(sp_list))
    
    d1 = {n:{} for n in nodos}
      
    for s1 in list(d1.keys()):
        n1 = sp_list[s1]
#         print('n1 = ', n1,'s1 = ',s1)
        for s2 in list(d1.keys()):
            n2 = sp_list[s2]
#             print('n2 = ',n2,'s2 = ',s2)
#             print('from: ',n1,' to ',n2)
            if (__can_combine(n1,n2)) and (__combine_spectr(n1,n2) in spectr):
#                 print(__combine_spectr(n1,n2),' is in spectre')
                edges_count = spectr.count(__combine_spectr(n1,n2)) #Obtenemos el número de ramas que van de un nodo a otro
                cont = 0
                d1[s1].update({s2:{}})
                while cont < edges_count:
                    
                    d1[s1][s2].update({cont:1}) #Actualizamos el diccionario de nodo destino con el número de ramas correspondiente
                    cont += 1
#                 print('Estado del dict: ',d1)
#                     ramas_x_vuelta+=1
        
#             ramas_x_vuelta = 0

        
    return d1
    
dir_graph = spectrum_2_graph(s)

print('\nMultigrafo dirigido: ',dir_graph)


# In[13]:


def path_2_sequence(l_path, spectrum_2):

    if len(l_path) == 0:
        return "-- Lista de caminos vacía, por lo tanto no se puede reconstruir la cadena --"
#     print('\nPath: ',l_path,'\nSpectro: ',spectrum_2)
    seq = spectrum_2[l_path[0][0]]
    
    for tup in l_path:
        seq = seq+spectrum_2[tup[1]][-1]
    
    return seq


def check_sequencing(len_seq, len_read):
    
    assert len_read > 1, "No se puede pasar 1 o menos como argumento de 'len_read'"
    
    s = random_sequence(len_seq) #Se genera una secuencia aleatoria de tamaño len_seq
    print("Cadena generada: ",s)
    
    spec = spectrum(s, len_read)
    
    spec_2 = spectrum_2(spec)
    
#     print("l-espectro generado: ",spec)
    
#     print("(l-1)-espectro: ",spec_2)

    d_mg = spectrum_2_graph(spec)
    
    print("Multigrafo generado: ",d_mg)
    
    camino = isthere_euler_path_directed_multigraph(d_mg)
    circuito = isthere_euler_circuit_directed_multigraph(d_mg)
    
    if (not camino) and (not circuito):
        print("No se puede reconstruir la secuencia dado que no hay camino euleriano")
        return False,0 #Si no hay camino euleriano, la cadena no se podrá reconstruir correctamente
    elif camino:
        path = euler_path_directed_multigraph(d_mg)
        
#         print("Camino euleriano: ",path)
        
        seq_2 = path_2_sequence(path, spec_2)
        
#         print(seq_2)
        
        return sorted(seq_2) == sorted(s)
    else:
        circuit = euler_circuit_directed_multigraph(d_mg)
        
        seq_2 = path_2_sequence(circuit,spec_2)
#         print(seq_2)
        return sorted(seq_2) == sorted(s)


# In[14]:


j = 0
k = 0
for i in range(50):
    print('\nCheck: ',i)
    result=check_sequencing(100, 3)
    if result:
        j += 1
    else:
        k += 1
            
print('Caminos correctos: ',j,'\nCircuitos: ',k,'\nTotal: ',j+k)

