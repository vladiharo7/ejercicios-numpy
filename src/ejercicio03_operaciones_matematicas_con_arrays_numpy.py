'''
Crea un array NumPy bidimensional de 4x5 con números aleatorios entre 0 y 100. Luego, realiza las siguientes operaciones:

Calcula la media de todo el array
Calcula la desviación estándar de todo el array
Encuentra el valor máximo y mínimo de todo el array
Calcula la suma de cada fila del array
Calcula la media de cada columna del array
Guarda cada resultado en variables separadas llamadas: media_total, desviacion_estandar, valor_maximo, valor_minimo, suma_filas y media_columnas.
'''

import numpy as np

# Crear un array bidimensional de 4x5 con números aleatorios entre 0 y 100
array = np.random.randint(0, 101, size=(4, 5))

# Calcular la media de todo el array
media_total = np.mean(array)

# Calcular la desviación estándar de todo el array
desviacion_estandar = np.std(array)

# Encontrar el valor máximo y mínimo de todo el array
valor_maximo = np.max(array)
valor_minimo = np.min(array)

# Calcular la suma de cada fila del array
suma_filas = np.sum(array, axis=1)

# Calcular la media de cada columna del array
media_columnas = np.mean(array, axis=0)

# Imprimir los resultados
print(f"Array:\n {array}")
print(f"Media total: {media_total}")
print(f"Desviación estándar: {desviacion_estandar}")
print(f"Valor máximo: {valor_maximo}")
print(f"Valor mínimo: {valor_minimo}")
print(f"Suma de cada fila: {suma_filas}")
print(f"Media de cada columna: {media_columnas}")

# --- IGNORE ---
