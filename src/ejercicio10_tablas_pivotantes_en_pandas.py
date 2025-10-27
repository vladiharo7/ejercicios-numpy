'''
Utilizando Pandas, crea una tabla pivotante que analice datos de ventas por categoría de producto y región.

Primero, crea un DataFrame con los siguientes datos:

'categoría': Distribuye 30 valores entre ['Electrónica', 'Ropa', 'Hogar']
'región': Distribuye 30 valores entre ['Norte', 'Sur', 'Este', 'Oeste']
'ventas': 30 valores aleatorios entre 100 y 1000
'unidades': 30 valores aleatorios entre 1 y 20
Luego, crea una tabla pivotante que muestre:

La suma total de ventas para cada combinación de categoría y región
Incluye totales por fila y columna (usando el parámetro margins)
Reemplaza los valores NaN con ceros
Tu solución debe importar las bibliotecas necesarias, crear el DataFrame de ejemplo y generar la tabla pivotante según las especificaciones.
'''
import pandas as pd
import numpy as np

# Semilla para reproducibilidad
np.random.seed(42)

# 1️⃣ Crear el DataFrame con 30 registros
data = {
    'categoría': np.random.choice(['Electrónica', 'Ropa', 'Hogar'], size=30),
    'región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], size=30),
    'ventas': np.random.randint(100, 1001, size=30),
    'unidades': np.random.randint(1, 21, size=30)
}

df = pd.DataFrame(data)

# 2️⃣ Crear la tabla pivotante
tabla_pivot = pd.pivot_table(
    df,
    values='ventas',
    index='categoría',
    columns='región',
    aggfunc='sum',
    margins=True,          # Totales por fila y columna
    margins_name='Total'   # Nombre personalizado para los totales
)

# 3️⃣ Reemplazar NaN por ceros
tabla_pivot.fillna(0, inplace=True)

# 4️⃣ Mostrar resultados
print("📊 Tabla pivotante: Ventas por categoría y región")
print(tabla_pivot)
