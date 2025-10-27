'''
Dado un DataFrame con información de ventas mensuales, calcula las siguientes estadísticas:

Utiliza el método describe() para obtener un resumen estadístico completo del DataFrame.
Calcula la media, mediana y desviación estándar de la columna 'ventas'.
Encuentra el valor máximo y mínimo de la columna 'unidades'.
Calcula la correlación entre las columnas 'ventas' y 'unidades'.
Para empezar, crea un DataFrame con los siguientes datos:

import pandas as pd
import numpy as np

# Datos de ventas mensuales
data = {
    'mes': ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo'],
    'ventas': [15200, 14800, 16700, 17500, 18200],
    'unidades': [120, 115, 140, 150, 160],
    'gastos': [5100, 4800, 5400, 5800, 6000]
}

# Crea el DataFrame
df_ventas = pd.DataFrame(data)
Tu código debe imprimir los resultados de cada estadística solicitada.
'''

import pandas as pd
import numpy as np

# Datos de ventas mensuales
data = {
    'mes': ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo'],
    'ventas': [15200, 14800, 16700, 17500, 18200],
    'unidades': [120, 115, 140, 150, 160],
    'gastos': [5100, 4800, 5400, 5800, 6000]
}

# Crea el DataFrame
df_ventas = pd.DataFrame(data)
print("\n Data Frame")
print(df_ventas)

# 1️⃣ Resumen estadístico completo
print("📊 Resumen estadístico con describe():")
print(df_ventas.describe())

# 2️⃣ Estadísticas de la columna 'ventas'
media_ventas = df_ventas['ventas'].mean()
mediana_ventas = df_ventas['ventas'].median()
desviacion_ventas = df_ventas['ventas'].std()

print("\n📈 Estadísticas de 'ventas':")
print(f"Media: {media_ventas}")
print(f"Mediana: {mediana_ventas}")
print(f"Desviación estándar: {desviacion_ventas}")

# 3️⃣ Máximo y mínimo de la columna 'unidades'
max_unidades = df_ventas['unidades'].max()
min_unidades = df_ventas['unidades'].min()

print("\n📌 Valores extremos de 'unidades':")
print(f"Máximo: {max_unidades}")
print(f"Mínimo: {min_unidades}")

# 4️⃣ Correlación entre 'ventas' y 'unidades'
correlacion = df_ventas['ventas'].corr(df_ventas['unidades'])

print("\n🔗 Correlación entre 'ventas' y 'unidades':")
print(f"Coeficiente de correlación: {correlacion}")
