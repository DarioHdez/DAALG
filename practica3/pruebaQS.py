import numpy as np
import math
import timeit
from random import shuffle
from statistics import median


# In[2]:


def _closest_pow2(n):
    return 2**(math.ceil(math.log(n, 2)))

def _fft_rec(t,K):
    #ret = []
    t = np.asarray(t)
    n = t.shape[0]

    if K == 0:
        return t
    else:
        evens, odds = t[::2],t[1::2]
        rec_evens = _fft_rec(evens,K-1)
        rec_odds = _fft_rec(odds,K-1)
        #a_len = int(2**(K-1))

        factor = np.exp(-2j * np.pi * np.arange(n) / n)

        return np.concatenate([rec_evens + factor[:n // 2] * rec_odds,
                               rec_evens + factor[n // 2:] * rec_odds])

def fft(t):
    t_len = len(t)

    next_pow = _closest_pow2(t_len)
    k = math.log(next_pow,2)
    num_add_zeros = next_pow - t_len
    zeros_array = np.array([0]*num_add_zeros)
    complete_array = np.concatenate((t,zeros_array),axis=0)


    return _fft_rec(complete_array,k)


def invert_fft(t, fft_func=fft):
    t_len = len(t)

    next_pow = _closest_pow2(t_len)
    k = math.log(next_pow,2)
    num_add_zeros = next_pow - t_len
    zeros_array = np.array([0]*num_add_zeros)
    complete_array = np.concatenate((t,zeros_array),axis=0)

    t_conj = [np.conj(n) for n in complete_array]

    dft = fft_func(t_conj)

    dft_conj = [np.conj(n) for n in dft]

    return np.array([n/len(t) for n in dft_conj])


# In[3]:


'''a = np.array([1,2,3,4])

s = fft(a)
#t = invert_fft(a)

print(s)


# In[4]:


b = list(np.random.randint(low=0, high=5, size=15))
b.append(10)
b'''


# In[5]:


def rand_polinomio(long=2**10,base=10):
    ret = list(np.random.randint(low=0, high=base, size=long-1).astype('uint8'))
    #print(ret)
    ret.append(np.random.randint(low=1, high=base))
    return ret

def poli_2_num(l_pol,base=10):
    ret = 0
    l_pol = list(np.array(l_pol).astype('uint8'))
    for i in range(len(l_pol)):
        ret += l_pol[i]*pow(base, i)
    return ret


#l = rand_polinomio()
#print(l)
#print(poli_2_num(l, 2))


# In[6]:


def rand_numero(num_digits,base=10):
    poli = rand_polinomio(num_digits, base)

    return poli_2_num(poli, base)

def num_2_poli(num,base=10):

    poli = []
    while True:
        remain = num % base
        poli.append(remain)
        num = num // base
        if num == 0:
            break

    return poli


#print(rand_numero(5))
#print(num_2_poli(1801, 5))


# In[7]:


def _padding_polinomios(l_pol_1, l_pol_2):
    long_prod = len(l_pol_1)+len(l_pol_2)-1 #longitud del polinomio producto

    next_pow2 = _closest_pow2(long_prod)

    num_add_zeros_pol1 = next_pow2 - len(l_pol_1)
    zeros_array_pol1 = np.array([0]*num_add_zeros_pol1)
    complete_array_pol1 = np.concatenate((l_pol_1,zeros_array_pol1),axis=0)

    num_add_zeros_pol2 = next_pow2 - len(l_pol_2)
    zeros_array_pol2 = np.array([0]*num_add_zeros_pol2)
    complete_array_pol2 = np.concatenate((l_pol_2,zeros_array_pol2),axis=0)

    return complete_array_pol1, complete_array_pol2

def mult_polinomios(l_pol_1, l_pol_2):
    prod = [0]*((len(l_pol_1)+len(l_pol_2))-1)

    for i in range(len(l_pol_1)):
        for j in range(len(l_pol_2)):
            prod[i+j] = prod[i+j] + l_pol_1[i] * l_pol_2[j]

    next_pow2 = _closest_pow2(len(prod))
    num_add_zeros_pol1 = next_pow2 - len(prod)
    zeros_array_pol1 = np.array([0]*num_add_zeros_pol1)
    prod = np.concatenate((prod,zeros_array_pol1),axis=0)

    return np.array(prod)

def mult_polinomios_fft(l_pol_1, l_pol_2, fft_func=fft):
    ret = []

    l_pol_1_pad, l_pol_2_pad = _padding_polinomios(l_pol_1, l_pol_2)

    # Realizamos la fft para pol_1 y pol_2
    fft_pol_1 = fft_func(l_pol_1_pad)
    fft_pol_2 = fft_func(l_pol_2_pad)

    if len(fft_pol_1) != len(fft_pol_2):
        print("Los tamaños son diferentes")
        pass
    else:
        for (i,j) in zip(fft_pol_1,fft_pol_2):
            ret.append(i*j)

    prod_pol = [n.real for n in invert_fft(ret, fft_func)]

    return np.array(prod_pol)


def mult_numeros(num1, num2):
    pol1 = num_2_poli(num1)
    pol2 = num_2_poli(num2)

    pol_mul = mult_polinomios(pol1, pol2)

    return poli_2_num(pol_mul)


def mult_numeros_fft(num1, num2, fft_func=fft):
    pol1 = num_2_poli(num1)
    pol2 = num_2_poli(num2)


    pol_mul = mult_polinomios_fft(pol1, pol2, fft_func)

    return poli_2_num(pol_mul)

'''
l_a = [1,2,5]
l_b = [1,2,3]
print(mult_polinomios(l_a, l_b))
print(mult_polinomios_fft(l_a,l_b))
print(mult_numeros(304, 509))
print(mult_numeros_fft(304, 509))
'''


# In[8]:


def time_mult_numeros(n_pairs, num_digits_ini, num_digits_fin, step):

    def local_rand_numeros(num):
        return mult_numeros(rand_numero(num), rand_numero(num))

    num_pair = 0
    times = []

    while num_pair < n_pairs:
        num_digits_actual = num_digits_ini

        while num_digits_actual <= num_digits_fin:
            # , setup = "from __main__ import mult_numeros, rand_numero, num_digits_actual", number = 1
            times.append(timeit.timeit(lambda: local_rand_numeros(num_digits_actual), number=1))
            num_digits_actual += step

        num_pair += 1

    return np.array(times)


def time_mult_numeros_fft(n_pairs, num_digits_ini, num_digits_fin, step, fft_func=fft):

    def local_rand_numeros(num):
        return mult_numeros_fft(rand_numero(num), rand_numero(num),fft_func=fft_func)

    num_pair = 0
    times = []

    while num_pair < n_pairs:
        num_digits_actual = num_digits_ini

        while num_digits_actual <= num_digits_fin:
            # , setup = "from __main__ import mult_numeros, rand_numero, num_digits_actual", number = 1
            times.append(timeit.timeit(lambda: local_rand_numeros(num_digits_actual), number=1))
            num_digits_actual += step

        num_pair += 1

    return np.array(times)

'''
print(time_mult_numeros(3, 3, 6, 1))
print('\n')
print(time_mult_numeros_fft(3, 3, 6, 1))
'''

a = [1, 2, 3]
b = [0,1,0.5]

print(mult_polinomios(a,b))

'''
num1 = rand_numero(30)
num2 = rand_numero(30)
        
prod_s = mult_numeros(num1, num2)
prod_f = mult_numeros_fft(num1, num2)

print(prod_s,"\n",prod_f,"\n",num1*num2)'''