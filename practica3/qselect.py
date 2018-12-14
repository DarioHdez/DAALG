import numpy as np
import math
import timeit
from random import shuffle
from statistics import median



def split(t, ini, fin):
    
    assert ini >= 0 and fin < len(t), "Los indices 'ini' y/o 'fin' se han introducido incorrectamente"
    
    pivot = t[ini]
    
    leftPointer, rightPointer = ini+1, fin
    
    while True:
        
        while leftPointer <= fin and t[leftPointer] <= pivot:
            leftPointer += 1
        
        while rightPointer >= ini and t[rightPointer] >= pivot:
            rightPointer -= 1
        
        if leftPointer > rightPointer:
            break;
        else:
            t[leftPointer], t[rightPointer] = t[rightPointer], t[leftPointer]
    
    t[ini], t[rightPointer] = t[rightPointer], t[ini]
    
    return rightPointer


def split_pivot(t, ini, fin, pivot=None):
    
    if (pivot == None):
        return split(t, ini, fin)
    
    assert ini >= 0 and fin < len(t), "Los indices 'ini' y/o 'fin' se han introducido incorrectamente"
    assert pivot in t, "pivot no está en la lista"
    '''
    pivotIndex = list(t).index(pivot)

    np.array(t)

    t[pivotIndex], t[fin] = t[fin], t[pivotIndex]  # Movemos el pivote al final
    index = ini
    for i in range(ini, fin):
        if t[i] < pivot:
            t[index], t[i] = t[i], t[index]
            index += 1
    t[fin], t[index] = t[index], t[fin]  # Movemos el pivote a su respectivo lugar
    
    return index
    '''
    z = ini

    while(z <= fin):
        if t[z] == pivot:
            i = 0
            break
        else:
            i = 1
        z = z + 1

    if i == 1:
        return pivot

    pivot = z
    t[ini], t[pivot] = t[pivot], t[ini]

    i = ini
    for j in range(ini+1, fin+1):
        if t[j] < t[ini]:
            i += 1
            t[i], t[j] = t[j], t[i]

    t[i], t[ini] = t[ini], t[i]

    return i



def qselect(t, ini, fin, ind, pivot=None):
    
    if ini == fin:
        return t[ini]
    
    split_p = split_pivot(t, ini, fin, pivot)
    
    l = split_p - ini +1
    
    if l == ind:
        return t[split_p]
    elif ind < l:
        return qselect(t, ini, split_p-1, ind, pivot)
    else:
        return qselect(t, split_p+1, fin, ind -l, pivot)
    

def qselect_sr(t, ini, fin, ind, pivot=None):
    
    if ini == fin:
        return t[ini]
    
    while True:
    
        split_p = split_pivot(t, ini, fin, pivot)

        l = split_p - ini+1

        if l == ind:
            return t[split_p]
        elif ind < l:
            fin = split_p-1
        else:
            ini = split_p+1
            ind -= l


                
def pivot_5(t, ini, fin):
    t = t[ini:fin+1]
    
    mid_lists = [t[i:i+5] for i in range(0, len(t), 5)]
    medians = [math.ceil(median(x)) for x in mid_lists]
    
    return math.ceil(median(medians))
    
    

def qselect_5(t, ini, fin, pos):
    
    if ini == fin:
        return t[ini]
    
    while True:
        
        pivotIndex = pivot_5(t, ini, fin)
        pivotIndex = split_pivot(t, ini, fin, pivotIndex)

        l = pivotIndex - ini + 1

        if l == pos:
            return t[pivotIndex]
        elif pos < l:
            fin = pivotIndex-1
        else:
            ini = pivotIndex+1
            pos -= l
    '''while True:
        if ini == fin:
            return ini
            
        pivotIndex = pivot_5(t, ini, fin)
        pivotIndex = split_pivot(t, ini, fin, pivotIndex)
        if pos == pivotIndex:
            return pos
        elif pos < pivotIndex:
            fin = pivotIndex - 1
        else:
            ini = pivotIndex + 1'''
        #return qselect_sr(t, ini, fin, pos)


#import random
#random.seed(2)
#a = [n for n in range(10)]
#shuffle(a)
#print(split(a, 0, 5))
#print(a)
#print(split_pivot(a, 0, 5))
#print(a)
#print(qselect_sr(a, 0, 9, 3))
#print(pivot_5(a, 0, 9))
#print(qselect_5(a,0,9,3))


# In[10]:


def qsort_5(t, ini, fin):

    print(len(t))
    
    if ini < fin:
        
        pivotIndex = pivot_5(t, ini, fin)
        pivotIndex = split_pivot(t, ini, fin, pivotIndex)
        
        qsort_5(t, ini, fin-1)
        qsort_5(t, ini+1, fin)


