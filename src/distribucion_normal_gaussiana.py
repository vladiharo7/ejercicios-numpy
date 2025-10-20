'''Ejemplo de generación de números aleatorios siguiendo una 
distribución normal (gaussiana) utilizando NumPy.
Distribución normal estándar (media=0, desviación=1) 
y distribución normal con media y desviación específicas.'''

import numpy as np
import matplotlib.pyplot as plt

# Crear un generador
rng = np.random.default_rng(12345)

normal = rng.normal(size=5)
print(normal)  # [-0.84912249  0.87569   -0.25771369  0.86377127 -1.17474048]

# Normal con media=100 y desviación=15
alturas = rng.normal(loc=170, scale=7, size=1000)
print(f"Media: {alturas.mean():.2f}, Desviación: {alturas.std():.2f}")
# Media: 169.93, Desviación: 6.97

# Visualización de la distribución (histograma)

plt.hist(alturas, bins=30, alpha=0.7)
plt.axvline(alturas.mean(), color='red', linestyle='dashed', linewidth=1)
plt.title('Distribución de alturas simuladas (cm)')
plt.xlabel('Altura (cm)')
plt.ylabel('Frecuencia')
plt.show()  # Descomentar para mostrar el gráfico
