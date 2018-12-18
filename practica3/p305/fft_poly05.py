# coding=utf-8

'''
Fichero fft_poly.py: Fichero que contiene los métodos de la parte de FFT y multiplicación de polinomios y números

@author: Borja González Farías y Darío Adrián Hernández
'''


import numpy as np
import math
import timeit
from random import shuffle
from statistics import median




def _closest_pow2(n):
    """
    Método privado que devuelve la siguiente potencia de 2 de un número

    :param n: Número que, en este caso, representa la longitud inicial de una lista
    :return: La siguiente potencia de 2 de n
    """
    return 2**(math.ceil(math.log(n, 2)))


def _fft_rec(t,K):
    """
    Método privado que realiza el algoritmo recursivo de la FFT

    :param t: Lista completa (con el padding de 0s ya realizado) de números enteros
    :param K: Exponente de 2^K (siendo 2^K la longitud de t)
    :return: Lista con FFT aplicada
    """

    ret = []

    if K == 0:
        return t
    else:
        evens, odds = t[::2],t[1::2]
        rec_evens = _fft_rec(evens,K-1)
        rec_odds = _fft_rec(odds,K-1)
        a_len = int(2**(K-1))

        for i in range(len(t)):
            n =  rec_evens[i % a_len] + np.exp((2j*math.pi * i/2**K)) * rec_odds[i % a_len]
            ret.append(n)

        return ret


def fft(t):
    """
    Método que realiza el algoritmo de la FFT (con padding de 0s incluído)

    :param t: Lista completa de números enteros
    :return: Array con FFT aplicada
    """

    t_len = len(t)

    next_pow = _closest_pow2(t_len)
    k = math.log(next_pow,2)
    num_add_zeros = next_pow - t_len
    zeros_array = np.array([0]*num_add_zeros)
    complete_array = np.concatenate((t,zeros_array),axis=0)


    return np.array(_fft_rec(complete_array,k))


def invert_fft(t, fft_func=fft):
    """
    Método que raplica la inversa a la lista a la que se le aplica FFT

    :param t: Lista completa (con el padding de 0s ya realizado) de números enteros
    :param fft_func: Método que relizará el algoritmo de FFT
    :return: Lista con FFT-invertida aplicada
    """

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


def rand_polinomio(long=2**10,base=10):
    """
    Método que genera un polinomio de longitud y coeficientes aleatorios

    :param long: Longitud del polinomio
    :param base: El número máximo que podrá tener un coeficiente
    :return: Lista representativa del polinomio (coeficientes)
    """

    ret = list(np.random.randint(low=0, high=base, size=long-1).astype('uint8'))
    ret.append(np.random.randint(low=1, high=base))
    return ret

def poli_2_num(l_pol,base=10):
    """
    Método que convierte un polinomio a un número

    :param l_pol: Lista con los coeficientes del polinomio
    :param base: Valor por el que se sustituirán las incógnitas del polinomio
    :return: Número entero resultante de la sustitución de la base en el polinomio
    """

    ret = 0
    for i in range(len(l_pol)):
        ret += int(l_pol[i]*pow(base, i))
    return ret


def rand_numero(num_digits,base=10):
    """
    Método que genera un número aleatorio al generar un polinomio al azar
    y pasando éste último a un número

    :param num_digits: Número de dígitos que tendrá el número
    :param base: La base en la que el número será representado
    :return: Número entero aleatorio
    """
    poli = rand_polinomio(num_digits, base)

    return poli_2_num(poli, base)


def num_2_poli(num,base=10):
    """
    Método que genera un polinomio a partir de un número

    :param num: Número a convertir
    :param base: La base en la que el número se desglosa en coeficientes
    :return: Lista de coeficientes del polinomio resultante
    """
    poli = []
    while True:
        remain = num % base
        poli.append(int(remain))
        num = num // base
        if num == 0:
            break

    return poli


def _padding_polinomios(l_pol_1, l_pol_2):
    """
    Método privado (auxiliar para la multiplicación de polinomios usando FFT)
    que realiza el padding de 0s según la longitud del polinomio producto.

    :param l_pol_1: Lista representativa del primer polinomio
    :param l_pol_2: Lista representativa del segundo polinomio
    :return: Las listas completas (con 0s) de ambos polinomios
    """
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
    """
    Método que multiplica dos polinomios de forma estándar

    :param l_pol_1: Lista representativa del primer polinomio
    :param l_pol_2: Lista representativa del segundo polinomio
    :return: Lista representativa del polinomio producto
    """

    prod = [0]*((len(l_pol_1)+len(l_pol_2))-1)

    for i in range(len(l_pol_1)):
        for j in range(len(l_pol_2)):
            prod[i+j] = prod[i+j] + l_pol_1[i] * l_pol_2[j]

    return prod

def mult_polinomios_fft(l_pol_1, l_pol_2, fft_func=fft):
    """
    Método que multiplica dos polinomios utilizando FFT

    :param l_pol_1: Lista representativa del primer polinomio
    :param l_pol_2: Lista representativa del segundo polinomio
    :param fft_func: Método FFT a utilizar
    :return: Lista representativa del polinomio producto
    """

    ret = []
    l_pol_1_pad, l_pol_2_pad = _padding_polinomios(l_pol_1, l_pol_2)

    # Realizamos la fft para pol_1 y pol_2
    fft_pol_1 = fft_func(l_pol_1_pad)
    fft_pol_2 = fft_func(l_pol_2_pad)

    ret = [i*j for i,j in zip(fft_pol_1, fft_pol_2)]

    prod_pol_np = np.rint(np.real(invert_fft(ret, fft_func)))

    prod_pol = [int(n) for n in prod_pol_np]
    

    return prod_pol


def mult_numeros(num1, num2):
    """
    Método que multiplica dos números a través de la multiplicación de polinomios

    :param num1: Primer número
    :param num2: Segundo número
    :return: Número resultante de convertir el polinomio producto a número
    """

    pol1 = num_2_poli(num1)
    pol2 = num_2_poli(num2)

    pol_mul = mult_polinomios(pol1, pol2)

    return poli_2_num(pol_mul)


def mult_numeros_fft(num1, num2, fft_func=fft):
    """
    Método que multiplica dos números a través de la multiplicación de polinomios con FFT

    :param num1: Primer número
    :param num2: Segundo número
    :param fft_func: Método FFT a utilizar
    :return: Número resultante de convertir el polinomio producto a número
    """

    pol1 = num_2_poli(num1)
    pol2 = num_2_poli(num2)


    pol_mul = mult_polinomios_fft(pol1, pol2, fft_func)

    return poli_2_num(pol_mul)


def time_mult_numeros(n_pairs, num_digits_ini, num_digits_fin, step):
    """
    Método que mide el tiempo de multiplicación estándar de varios números

    :param n_pairs: Número de parejas de números a multiplicar entre sí
    :param num_digits_ini: Número de dígitos inicial de cada número
    :param num_digits_fin: Número de dígitos final de cada número
    :param step: El número en que se incrementará el número de dígitos en cada vuelta
    :return: Lista de los tiempos de cada multiplicacíon
    """

    def local_rand_numeros(num):
        """
        Método local que multiplica dos números aleatorios de forma estándar

        :param num: Número de dígitos de ambos números
        :return: Número producto
        """

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
    """
    Método que mide el tiempo de multiplicación FFT de varios números

    :param n_pairs: Número de parejas de números a multiplicar entre sí
    :param num_digits_ini: Número de dígitos inicial de cada número
    :param num_digits_fin: Número de dígitos final de cada número
    :param step: El número en que se incrementará el número de dígitos en cada vuelta
    :param fft_func: Método FFT a utilizar
    :return: Lista de los tiempos de cada multiplicacíon
    """

    def local_rand_numeros(num):
        """
        Método local que multiplica dos números aleatorios de usando FFT

        :param num: Número de dígitos de ambos números
        :return: Número producto
        """

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