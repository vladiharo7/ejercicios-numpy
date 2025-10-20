'''Ejemplo que ilustra el uso de las convenciones de importación en un flujo de trabajo típico
   y compara el rendimiento de diferentes métodos de creación de arrays en NumPy.'''

import time
import numpy as np

# Comparación de rendimiento: lista vs array
N = 10000000

# Crear lista grande y luego convertir
inicio = time.time()
lista_grande = [i for i in range(N)]
arr_desde_lista = np.array(lista_grande)
fin = time.time()
print(f"Tiempo lista→array: {fin - inicio:.4f} segundos")

# Crear array directamente
inicio = time.time()
arr_directo = np.arange(N)
fin = time.time()
print(f"Tiempo array directo: {fin - inicio:.4f} segundos")

# Cargar desde archivo vs crear en memoria
arr_grande = np.arange(N)
np.save('arr_grande.npy', arr_grande)

inicio = time.time()
arr_cargado = np.load('arr_grande.npy')
fin = time.time()
print(f"Tiempo carga desde archivo: {fin - inicio:.4f} segundos")
