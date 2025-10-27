import pandas as pd
import numpy as np

def separator_line():
    print("\n" + "-"*50 + "\n")

# Creamos un DataFrame de ejemplo
df = pd.DataFrame({
    'numérico': [10, 20, 30, 40, 50, np.nan],
    'categórico': ['a', 'b', 'c', 'a', 'b', 'c'],
    'fecha': pd.date_range('20230101', periods=6)
})

# Aplicamos describe() al DataFrame
resumen = df.describe()
print(resumen)

separator_line()

# Incluir todos los tipos de datos
resumen_completo = df.describe(include='all')
print(resumen_completo)

# Especificar percentiles personalizados
resumen_percentiles = df.describe(percentiles=[0.2, 0.4, 0.6, 0.8])
print(resumen_percentiles)

separator_line()

# Solo columnas numéricas (comportamiento por defecto)
df.describe(include=[np.number])

# Solo columnas de texto
df.describe(include=['object'])

# Solo columnas de fecha
df.describe(include=['datetime64'])

# Combinación de tipos
df.describe(include=['object', 'datetime64'])

separator_line()

# Aplicar describe() a una Serie numérica
serie_numerica = df['numérico']
print(serie_numerica.describe())

# Aplicar describe() a una Serie categórica
serie_categorica = df['categórico']
print(serie_categorica.describe())


separator_line()

# Cargar un conjunto de datos
# df_ventas = pd.read_csv('ventas_mensuales.csv')

# Obtener un resumen estadístico
# resumen_ventas = df_ventas.describe()
# print(resumen_ventas)

separator_line()
# Analizar si hay valores extremos comparando min/max con los cuartiles
def verificar_outliers(df):
    stats = df.describe()
    for columna in stats.columns:
        q1 = stats.loc['25%', columna]
        q3 = stats.loc['75%', columna]
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        
        if stats.loc['min', columna] < limite_inferior or stats.loc['max', columna] > limite_superior:
            print(f"Posibles outliers en columna {columna}")

# verificar_outliers(df_ventas)

separator_line()

import matplotlib.pyplot as plt

# Visualizar estadísticas descriptivas
def visualizar_estadisticas(df, columna):
    stats = df[columna].describe()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de caja (boxplot)
    df[columna].plot(kind='box', ax=ax1)
    ax1.set_title(f'Distribución de {columna}')
    
    # Gráfico de barras para estadísticas
    stats.drop('count').plot(kind='bar', ax=ax2)
    ax2.set_title(f'Estadísticas de {columna}')
    
    plt.tight_layout()
    plt.show()

# visualizar_estadisticas(df_ventas, 'ventas')
separator_line()

# Crear un DataFrame de ventas más completo
np.random.seed(42)
df_ventas = pd.DataFrame({
    'fecha': pd.date_range('20230101', periods=100),
    'tienda': np.random.choice(['T1', 'T2', 'T3', 'T4'], 100),
    'producto': np.random.choice(['P1', 'P2', 'P3'], 100),
    'vendedor': np.random.choice(['V1', 'V2', 'V3', 'V4', 'V5'], 100),
    'unidades': np.random.randint(1, 20, 100),
    'precio_unitario': np.random.uniform(10, 100, 100).round(2)
})

# Calcular el importe total de cada venta
df_ventas['importe'] = df_ventas['unidades'] * df_ventas['precio_unitario']

# 1. Ventas totales por tienda y producto
ventas_por_tienda_producto = df_ventas.groupby(['tienda', 'producto'])['importe'].sum().unstack(fill_value=0)
print("Ventas por tienda y producto:\n", ventas_por_tienda_producto)

# 2. Estadísticas de ventas por vendedor
stats_vendedor = df_ventas.groupby('vendedor').agg({
    'importe': ['sum', 'mean', 'count'],
    'unidades': ['sum', 'mean']
})
print("\nEstadísticas por vendedor:\n", stats_vendedor)

# 3. Evolución temporal: ventas semanales
df_ventas['semana'] = df_ventas['fecha'].dt.isocalendar().week
ventas_semanales = df_ventas.groupby('semana')['importe'].agg(['sum', 'mean', 'count'])
print("\nVentas semanales:\n", ventas_semanales)

# 4. Identificar el mejor producto por tienda
mejor_producto = df_ventas.groupby(['tienda', 'producto'])['importe'].sum().groupby(level=0).idxmax()
print("\nMejor producto por tienda:\n", mejor_producto)

separator_line()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Creamos un DataFrame con variables relacionadas
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'edad': np.random.randint(18, 70, n),
    'ingresos': np.random.normal(30000, 15000, n),
    'gastos': np.random.normal(25000, 10000, n),
    'experiencia': np.random.randint(0, 40, n),
    'satisfaccion': np.random.randint(1, 11, n)
})

# Añadimos algunas relaciones
df['ingresos'] = df['ingresos'] + df['edad'] * 500 + np.random.normal(0, 5000, n)
df['gastos'] = df['ingresos'] * 0.7 + np.random.normal(0, 5000, n)
df['satisfaccion'] = 10 - 0.1 * (df['gastos'] / df['ingresos'] * 10) + np.random.normal(0, 2, n)

separator_line()

# Calcular la matriz de correlación
matriz_corr = df.corr()
print("Matriz de correlación:")
print(matriz_corr)


separator_line()

# Correlación de Spearman (basada en rangos, detecta relaciones monótonas no lineales)
corr_spearman = df.corr(method='spearman')
print("\nCorrelación de Spearman:")
print(corr_spearman)

# Correlación de Kendall (también basada en rangos, más robusta a outliers)
corr_kendall = df.corr(method='kendall')
print("\nCorrelación de Kendall:")
print(corr_kendall)

separator_line()

# Crear un mapa de calor (heatmap) de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Matriz de correlación')
plt.tight_layout()
plt.show()

separator_line()

# Crear un pairplot para visualizar relaciones bivariadas
sns.pairplot(df)
plt.suptitle('Relaciones entre variables', y=1.02)
plt.show()

