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
        
        while leftPointer < fin and t[leftPointer] < pivot:
            leftPointer += 1
        
        while rightPointer > ini and t[rightPointer] > pivot:
            rightPointer -= 1
        
        if leftPointer >= rightPointer:
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

    pivotIndex = ini

    while(pivotIndex <= fin):
        if t[pivotIndex] == pivot:
            notFoundFlag = 0
            break
        else:
            notFoundFlag = 1
        pivotIndex = pivotIndex + 1

    if notFoundFlag == 1:
        pivotIndex = ini

    pivot = pivotIndex
    t[ini], t[pivot] = t[pivot], t[ini]

    return split(t, ini, fin)



def qselect(t, ini, fin, ind, pivot=None):

    if ini == fin:
        return t[ini]
    split_p = split_pivot(t, ini, fin, pivot)
    
    l = split_p - ini + 1
    #print('Ele: ',l)
    if l == (ind+1):
        return t[split_p]
    elif ind < l:
        return qselect(t, ini, split_p-1, ind, pivot)
    else:
        return qselect(t, split_p+1, fin, ind - l, pivot)
    

def qselect_sr(t, ini, fin, ind, pivot=None):
    
    if ini == fin:
        return t[ini]
    
    while True:
    
        split_p = split_pivot(t, ini, fin, pivot)

        l = split_p - ini+1

        if l == (ind+1):
            return t[split_p]
        elif ind < l:
            fin = split_p-1
        else:
            ini = split_p+1
            ind -= l


                
def pivot_5(t, ini, fin):

    t = t[ini:fin+1]

    if len(t) == 0:
        return None
    
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

        if l == (pos+1):
            return t[pivotIndex]
        elif pos < l:
            fin = pivotIndex-1
        else:
            ini = pivotIndex+1
            pos -= l



def qsort_5(t, ini, fin):

    if ini < fin:
        
        pivotIndex = pivot_5(t, ini, fin)
        pivotIndex = split_pivot(t, ini, fin, pivotIndex)
        
        qsort_5(t, ini, pivotIndex-1)
        qsort_5(t, pivotIndex+1, fin)


