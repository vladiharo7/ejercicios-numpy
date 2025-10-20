''' Ejemplo de eficiencia de NumPy vs listas Python '''
import time
import numpy as np

# Operación con listas Python
python_list1 = list(range(9000000))
python_list2 = list(range(9000000))

start = time.time()
result_list = [x + y for x, y in zip(python_list1, python_list2)]
end = time.time()
print(f"Tiempo con listas Python: {end - start:.6f} segundos")

# Misma operación con arrays NumPy
numpy_array1 = np.array(range(9000000))
numpy_array2 = np.array(range(9000000))

start = time.time()
result_array = numpy_array1 + numpy_array2
end = time.time()
print(f"Tiempo con arrays NumPy: {end - start:.6f} segundos")
