#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string, random
import numpy as np
import sys
import time
import copy

import fft_poly as fp
import qselect  as qs


eps = 1.e-5

def check_pr_3():

    #################### fft_poly

    print("\nchecking fft e inv_fft ..........")
    
    # #### 1. Checking fft, invff and np.fft
    # 
    # * Generar aleatoriamente tabla de floats
    # * Comprobar que invert_fft(fp.fft(t) = t
    # * Comprobar que fp.fft(t) - np.fp.fft(t) = 0
    inv_fft_t = []
    for i in range(10):
        
        len_t = np.random.randint(100, 151)
        
        # random table: random shift + permutation
        t = np.random.randn(len_t)
        
        # fft(t) vs np.fft(t)
        pot = np.random.randint(5, 9)
        len_t = int(2.**pot)
        t = np.random.randn(len_t)
        
        fft_t = fp.fft(t)
        np_fft_t = np.fft.fft(t)
        if np.linalg.norm(fft_t[ 1 : ] - np_fft_t[ 1 : ][ : : -1]) > eps:
            print("iter", i)
            print("error en fft:\n", fft_t, "\n", np_fft_t)
            break
        
        # t vs invert.fft(fft(t))
        fft_t = fp.fft(t)
        inv_fft_t = fp.invert_fft(fft_t)
        if np.linalg.norm(t - inv_fft_t[ : len(t)]) > eps:
            print("iter", i)
            print("error en fft:\n", t, "\n", inv_fft_t)
            break
        
    print("last values of the first elements of t, fft and inv_fft")    
    print(t[  : 5])
    print(fft_t[  : 5])
    print(inv_fft_t[  : 5])
    
    print("ok") 
    
    _ = input("pulsar Intro para continuar ....................\n")
    
    
    print("\nchecking mult de polinomios ..........")
       
    # #### 2. Checking mult_polis estandar y fft, invff and np.fft
    # 
    # * Generar aleatoriamente polinomios
    # * Comprobar que mult fft y estandar coinciden
    
    for i in range(10):
        
        pot = np.random.randint(5, 9)
        base = np.random.randint(2, 11)
        
        p1 = fp.rand_polinomio(int(2**pot), base=base)
        p2 = fp.rand_polinomio(int(2**pot), base=base)
    
        prod_s = fp.mult_polinomios(p1, p2)
        prod_f = fp.mult_polinomios_fft(p1, p2)
        
        if np.linalg.norm(prod_s - prod_f[ : len(prod_s)]) > 0.:
            print("iter", i)
            print("error en prod:\n", prod_s, "\n", prod_f)
            break    
    
    print("last values of the first elements of p1, p2, prod_standard, prod_ffft")
    print(p1[  : 5])   
    print(p2[  : 5])
    print(prod_s[  : 5])
    print(prod_f[  : 5])
    
    print("ok") 
    
    _ = input("pulsar Intro para continuar ....................\n")
    
    
    print("\nchecking mult de números ..........")
    
    # #### 3. Checking mult_numeros estandar, fft y python
    # 
    # * Generar aleatoriamente polinomios
    # * Comprobar que mult fft, estandar y python coinciden
    
    for i in range(10):
        
        num_d = np.random.randint(50, 101)
        base  = np.random.randint(4, 11)
        
        num1 = fp.rand_numero(num_d, base=base)
        num2 = fp.rand_numero(num_d, base=base)
        
        prod_s = fp.mult_numeros(num1, num2)
        prod_f = fp.mult_numeros_fft(num1, num2)
        prod_p = num1 * num2
        
        if prod_s != prod_f:
            print("iter", i)
            print("error en prod s o f:\n", prod_s, "\n", prod_f)
            break    
        
        if prod_s != prod_p:
            print("iter", i)
            print("error en prod s:\n", prod_s, "\n", prod_f)
            break    
    
    print("base", base, "num_digits", num_d)
    print("last values of num1, num2, prod_standard, prod_fft and prod_python")
    print(num1)
    print(num2)
    print(prod_s)
    print(prod_f)
    print(prod_p)
    
    print("ok") 
    
    #################### qselect
    
    _ = input("pulsar Intro para continuar ....................\n")
    
    
    print("\nchecking qselect y qselect_5 ..........")
    
    # #### 1. Checking qselect, qselect5:
    # 
    # * Generar aleatoriamente tabla, shift y pos a buscar en shift + tabla.
    # * Comprobar que el valor devuelto - shift coincide con pos.
    
    for i in range(150):
        len_list = np.random.randint(100, 151)
    
        # random list: random shift + permutation
        shf = np.random.randint(-len_list//2, len_list//2)
        t = shf + np.random.permutation(len_list)
        
        # random search limits
        ini = np.random.randint(len_list//2)
        fin = np.random.randint(len_list//2, len_list)
        
        # random pos to search
        pos = np.random.randint(fin-ini+1)
        
        # checking qselect
        val_pos = qs.qselect(t, ini, fin, pos)
        
        # correct value en pos
        #qs.qsort_5(t, ini, fin)
        print(i)
        val_ok  = t[ini+pos]
            
        if val_pos != val_ok:
            print("iter", i)
            print("error en qselect:", pos, ini, fin, val_pos)
            print(t, '\n')
            break
        
        # checking qselect_5
        t = shf + np.random.permutation(len_list)
        
        val_pos = qs.qselect_5(t, ini, fin, pos)
        
        qs.qsort_5(t, ini, fin)
        val_ok  = t[ini+pos]
        
        # break if value found by qselect5 not correct
        if val_pos != val_ok:
            print("error en qselect5:", pos, val_pos)
            print(t, '\n')
            break
    
    #print last table, position sought and value found
    print(t[ini : fin+1])
    print(pos, t[ini+pos], val_pos)
    
    print("ok") 
    
    _ = input("pulsar Intro para continuar ....................\n")
    
    
    print("\nchecking qsort_5 ..........")
    
    # #### 2. Checking quicksort5
    # 
    # * Generar aleatoriamente tabla, shift y pos a buscar en shift + tabla.
    # * Comprobar que el valor devuelto - shift coincide con pos.
    
    for _ in range(100):
        
        len_list = np.random.randint(100, 151)
        
        # rndom shifted table
        shf = np.random.randint(-len_list//2, len_list//2)
        t0  = shf + np.random.permutation(len_list)
        t   = copy.deepcopy(t0)
        
        # random search limits
        ini = np.random.randint(len_list//2)
        fin = np.random.randint(len_list//2, len_list)
        
        #qs.qsort_5(t, ini, fin, verbose=False)
        
        # check if ordered: break if not ordered
        if not np.all(t[ini : fin] <= t[ini+1 : fin+1]):
            print("error on \n", t0[ini : fin+1], '\n', t[ini : fin+1])   
            
    print("ok")

if __name__ == '__main__':
    #check_pr_2(sys.argv[1:])
    check_pr_3()
