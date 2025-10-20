'''
Crea los siguientes arrays de NumPy:

Un array unidimensional con los números del 1 al 10 utilizando array()
Una matriz de ceros de tamaño 3x3 utilizando zeros()
Un array unidimensional con 5 unos utilizando ones()
Un array con 8 valores equidistantes entre 0 y 1 (ambos incluidos) utilizando linspace()
Un array con los números pares del 2 al 20 utilizando arange()
Asegúrate de importar NumPy correctamente al inicio de tu código.
'''

import numpy as np

# Array unidimensional con los números del 1 al 10
array_1_10 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Array del 1 al 10:", array_1_10)
print('---------------------------------------------------')

# Matriz de ceros de tamaño 3x3
matriz_ceros = np.zeros((3, 3))
print("Matriz de ceros 3x3:\n", matriz_ceros)
print('---------------------------------------------------')

# Array unidimensional con 5 unos
array_unos = np.ones(5)
print("Array de 5 unos:", array_unos)
print('---------------------------------------------------')

# Array con 8 valores equidistantes entre 0 y 1
array_equidistantes = np.linspace(0, 1, 8)
print("Array con 8 valores equidistantes entre 0 y 1:", array_equidistantes)
print('---------------------------------------------------')

# Array con los números pares del 2 al 20
array_pares = np.arange(2, 21, 2)
print("Array con números pares del 2 al 20:", array_pares)
print('---------------------------------------------------')
