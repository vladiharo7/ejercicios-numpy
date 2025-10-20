'''Comparación de rendimiento entre reshape() y resize() en NumPy'''

import time
import numpy as np

# Creamos un array grande
TAMANIO = 10_000_000
arr = np.random.rand(TAMANIO)

# Medimos tiempo para reshape
inicio = time.time()
reshaped = arr.reshape(10000, 1000)
tiempo_reshape = time.time() - inicio

# Medimos tiempo para resize (método)
arr_copy = arr.copy()  # Hacemos copia para no modificar original
inicio = time.time()
arr_copy.resize(10000, 1000)
tiempo_resize_metodo = time.time() - inicio

# Medimos tiempo para np.resize (función)
inicio = time.time()
resized = np.resize(arr, (10000, 1000))
tiempo_resize_funcion = time.time() - inicio

print(f"Tiempo reshape(): {tiempo_reshape*1000:.2f} ms")
print(f"Tiempo resize() método: {tiempo_resize_metodo*1000:.2f} ms")
print(f"Tiempo np.resize() función: {tiempo_resize_funcion*1000:.2f} ms")
