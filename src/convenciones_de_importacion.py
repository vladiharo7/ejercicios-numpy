'''Importaciones siguiendo convenciones estándar'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(42)  # Para reproducibilidad
datos = np.random.normal(loc=0, scale=1, size=1000)

# Crear DataFrame
df = pd.DataFrame({
    'valores': datos,
    'valores_cuadrado': datos ** 2,
    'valores_abs': np.abs(datos)
})

# Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(df.describe())

# Visualización básica
plt.figure(figsize=(10, 6))
plt.hist(df['valores'], bins=30, alpha=0.7)
plt.title('Distribución de valores')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)
plt.show()
