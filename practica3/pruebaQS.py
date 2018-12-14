import numpy as np
import sys
from random import shuffle

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

def quickselect(A, left, right, k, pivot=None):

    if left == right:
        return A[left];

    pivotIndex = split_pivot(A, left, right, pivot);

    if k == pivotIndex:
        return A[k]

    elif k < pivotIndex:
        return quickselect(A, left, pivotIndex - 1, k);

    else:
        return quickselect(A, pivotIndex + 1, right, k);


A = [7,5,4,6,3,0,9,2,1,8,10,11]
k = 2

print("K'th smallest element is ",quickselect(A, 5, 11, k));



#[91,73,60,95,87,31,40,72,69,25,12,32,17,51,92,21,47,8,62,4,101,11,68,6,86,43,2,77,93,61,1,85,76,58,97,5,19,98,71,29,35,24,45,34,9,74,30,18,7,33,70,56,55,57,65,81,94,64,44,90,78,0,46,59,28,23,48,38,16,10,26,15,66,99,54,13,63,27,42,22,96,83,75,52,36,50,80,39,67,3,20,37,100,49,84,79,88,14,82,89,53,41]