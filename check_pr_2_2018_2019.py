#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string, random
import numpy as np
import copy
import sys

from sklearn.linear_model import LinearRegression

#sys.path.append(r"D:\Google Drive\Cursos\DAA\practicas\20162017\python")
import secuencias as sq

############################################################################## funciones auxiliares
    
############################################################################## main
#def check_pr_2(args):
def check_pr_2():
    """Prueba las funciones de secuencias.py desarrolladas en la práctica 2
    """
    
    np.set_printoptions(precision=3)

    #print("....................................................................................................")
    print(".................... checking euler circuit and sequence reconstruction ....................")
    
    ##### varios ejemplos de grafos en principio dirigidos
    # Añadir ejemplos que se hayan usado
    
    
    print("\ncomprobamos existencia de circuitos eulerianos ..........")
    d_g = { 0: {1: {0: 1}}, 1: {3: {0: 1}}, 2: {0: {0: 1}}, 3: {4: {0:1}}, 4: {9: {0:1}}, 5: {8: {0:1}}, 6: {5: {0: 1}}, 7: {6: {0:1}}, 8: {2: {0:1}}, 9: {7: {0:1}} }
    
    print("\ncalculamos tablas adj, inc ..........")
    print("adj_inc\n", sq.adj_inc_directed_multigraph(d_g) )
    
    if sq.isthere_euler_circuit_directed_multigraph(d_g):
        print("\nejemplo de paseo ..........")
        d_g_copy = copy.deepcopy(d_g)
        print( sq.euler_walk_directed_multigraph(0, d_g_copy) )
        
        print("\nencontramos circuitos eulerianos ..........")
        d_g_copy = copy.deepcopy(d_g)
        print( sq.euler_circuit_directed_multigraph(d_g_copy, 0) )
    
    else:
        print("\nalgo ha ido mal ..........")
    
    _ = input("pulsar Intro para continuar ....................\n")
    
    
    print("\ncomprobamos existencia de caminos eulerianos ..........")
    d_g = { 0: {1: {0: 1}}, 1: {3: {0: 1}}, 2: {0: {0: 1}}, 3: {4: {0:1}}, 4: {9: {0:1}}, 5: {8: {0:1}}, 6: {}, 7: {6: {0:1}}, 8: {2: {0:1}}, 9: {7: {0:1}} }
    
    print("\ncalculamos tablas adj, inc ..........")
    print("adj_inc\n", sq.adj_inc_directed_multigraph(d_g) )
    
    if sq.isthere_euler_path_directed_multigraph(d_g):
        print("\nvértices inicial y final ..........")
        u, v = sq.first_last_euler_path_directed_multigraph(d_g)
        print(u, v)
    
        print("\nejemplo de paseo ..........")
        d_g_copy = copy.deepcopy(d_g)
        print( sq.euler_walk_directed_multigraph(u, d_g_copy) )
        
        print("\nencontramos circuitos eulerianos ..........")
        d_g_copy = copy.deepcopy(d_g)
        print( sq.euler_path_directed_multigraph(d_g_copy) )
    
    else:
        print("\nalgo ha ido mal ..........")
    
    _ = input("pulsar Intro para continuar ....................\n")
    
        
    print("\ncomprobamos cálculos de espectros y grafos sobre secuencias sencillas ..........")
    #seq = "ACGT"
    seq = sq.random_sequence(4)
    print(seq)
    spec = sq.spectrum(seq, len_read=3)
    print(spec)
    spec_2 = sq.spectrum_2(spec)
    print(spec_2)
    d_g = sq.spectrum_2_graph(spec)
    print(d_g)
    
    _ = input("pulsar Intro para continuar ....................\n")

    print("comprobamos reconstrucción ..........")
    # generamos secuencia aleatoria y calculamos espectro y grafo
    len_seq = 100 + np.random.randint(50)
    len_read = 3
    print("len_seq", len_seq,  "len_read", len_read, '\n')
    
    seq = sq.random_sequence(len_seq)
    spec = sq.spectrum(seq, len_read)
    d_g  = sq.spectrum_2_graph(spec)
   
    # buscamos circuito euleriano y reconstruimos cadenas 
    e_path = sq.euler_path_directed_multigraph(d_g)
    spec_2 = sq.spectrum_2(spec)
    seq_rec = sq.path_2_sequence(e_path, spec_2)
    print('\t'+seq+'\n', '\t'+seq_rec+'\n')
    
    # comprobamos coherencia de la reconstrucción
    spec_rec = sq.spectrum(seq_rec, len_read)
    print("seq  iguales?", seq == seq_rec)
    print("spec iguales?", sorted(list(spec)) == sorted(list(spec_rec)))
    
    _ = input("pulsar Intro para continuar ....................\n")
    
    len_seq = 100 + np.random.randint(50)
    len_read = np.random.randint(10, 50)
    print("len_seq", len_seq,  "len_read", len_read, '\n')

    seq = sq.random_sequence(len_seq)
    print("SECUENCIA: ",seq)
    spec = sq.spectrum(seq, len_read)
    print("SPECTRO: ",spec)
    d_g  = sq.spectrum_2_graph(spec)

    print("GRAFO: ",d_g)
    
    # buscamos circuito euleriano y reconstruimos cadenas 
    e_path = sq.euler_path_directed_multigraph(d_g)
    print("CAMINO: ",e_path)
    spec_2 = sq.spectrum_2(spec)
    seq_rec = sq.path_2_sequence(e_path, spec_2)
    print('\t'+seq+'\n', '\t'+seq_rec+'\n')
    
    # comprobamos coherencia de la reconstrucción
    spec_rec = sq.spectrum(seq_rec, len_read)
    print("seq  iguales?", seq == seq_rec)
    print("spec iguales?", sorted(list(spec)) == sorted(list(spec_rec)))
    
  
if __name__ == '__main__':
    #check_pr_2(sys.argv[1:])
    check_pr_2()
    
