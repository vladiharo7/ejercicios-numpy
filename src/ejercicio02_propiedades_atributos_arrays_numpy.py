'''
Crea un programa que genere un array NumPy tridimensional de tamaño 4x3x2 
con valores aleatorios enteros entre 1 y 100. 
Luego, muestra por pantalla la siguiente información sobre el array:

La forma (shape) del array
El número de dimensiones (ndim) del array
El número total de elementos (size) del array
El tipo de datos (dtype) del array
El tamaño en bytes de cada elemento (itemsize) del array
El tamaño total en bytes (nbytes) del array
Finalmente, verifica e imprime si el tamaño total en bytes (nbytes) 
es igual al producto del número de 
elementos (size) por el tamaño de cada elemento (itemsize).
'''
import numpy as np

# Crear un generador de números aleatorios
rng = np.random.default_rng()

# Generar un array tridimensional de tamaño 4x3x2 con valores aleatorios enteros entre 1 y 100
array_3d = rng.integers(1, 101, size=(4, 3, 2))

# Mostrar la información solicitada sobre el array
print("Array generado:\n", array_3d)
print(f"Forma (shape) del array: {array_3d.shape}")
print(f"Número de dimensiones (ndim) del array: {array_3d.ndim}")
print(f"Número total de elementos (size) del array: {array_3d.size}")
print(f"Tipo de datos (dtype) del array: {array_3d.dtype}")
print(f"Tamaño en bytes de cada elemento (itemsize) del array: {array_3d.itemsize}")
print(f"Tamaño total en bytes (nbytes) del array: {array_3d.nbytes}")

# Verificar si nbytes es igual al producto de size por itemsize
calculated_nbytes = array_3d.size * array_3d.itemsize
print(f"¿nbytes es igual a size * itemsize?: {array_3d.nbytes == calculated_nbytes}")
