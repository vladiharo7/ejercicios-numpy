import pandas as pd
import numpy as np

def separator_line():
    print("\n" + "-"*50 + "n")

# Sintaxis básica
# df.pivot_table(
#     values='columna_valores',
#     index='columna_filas',
#     columns='columna_columnas',
#     aggfunc='función_agregación',
#     fill_value=0,
#     margins=False
# )


# Creamos un DataFrame de ejemplo
data = {
    'fecha': pd.date_range(start='2023-01-01', periods=20),
    'producto': np.random.choice(['Laptop', 'Tablet', 'Smartphone'], 20),
    'región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 20),
    'ventas': np.random.randint(5, 50, 20),
    'unidades': np.random.randint(1, 10, 20)
}

df = pd.DataFrame(data)

# Creamos una tabla pivotante básica
tabla_ventas = df.pivot_table(
    values='ventas',
    index='producto',
    columns='región',
    aggfunc='sum'
)

print(tabla_ventas)

separator_line()

# Tabla pivotante con múltiples índices y columnas
tabla_compleja = df.pivot_table(
    values=['ventas', 'unidades'],
    index=['producto'],
    columns=['región'],
    aggfunc={'ventas': 'sum', 'unidades': 'mean'},
    fill_value=0,
    margins=True
)

print(tabla_compleja)


separator_line()

# pd.crosstab(
#     index=df['columna_filas'],
#     columns=df['columna_columnas'],
#     values=df['columna_valores'],
#     aggfunc='función_agregación',
#     normalize=False,
#     margins=False
# )

# Tabla de contingencia básica
tabla_contingencia = pd.crosstab(
    index=df['producto'],
    columns=df['región']
)

print(tabla_contingencia)

separator_line()

# Tabla normalizada por filas
tabla_normalizada = pd.crosstab(
    index=df['producto'],
    columns=df['región'],
    normalize='index'
)

print(tabla_normalizada)

separator_line()

# Crosstab con valores y función de agregación
tabla_ventas_cruzadas = pd.crosstab(
    index=df['producto'],
    columns=df['región'],
    values=df['ventas'],
    aggfunc='sum',
    margins=True
)

print(tabla_ventas_cruzadas)


separator_line()

# Creamos datos más realistas
data_ventas = {
    'fecha': pd.date_range(start='2023-01-01', periods=100),
    'categoría': np.random.choice(['Electrónica', 'Ropa', 'Hogar', 'Deportes'], 100),
    'región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 100),
    'canal': np.random.choice(['Online', 'Tienda física'], 100),
    'ventas': np.random.randint(100, 1000, 100),
    'unidades': np.random.randint(1, 20, 100)
}

df_ventas = pd.DataFrame(data_ventas)

# Análisis de ventas totales por categoría y región
resumen_ventas = df_ventas.pivot_table(
    values='ventas',
    index='categoría',
    columns='región',
    aggfunc='sum',
    fill_value=0,
    margins=True
)

print("Resumen de ventas por categoría y región:")
print(resumen_ventas)


separator_line()

# Distribución de canales por región (normalizado)
distribución_canales = pd.crosstab(
    index=df_ventas['región'],
    columns=df_ventas['canal'],
    normalize='index'
)

print("\nDistribución de canales de venta por región (%):")
print(distribución_canales.round(2) * 100)


separator_line()

# Tabla pivotante con formato personalizado
tabla_formateada = df_ventas.pivot_table(
    values=['ventas', 'unidades'],
    index=['categoría'],
    columns=['región'],
    aggfunc={'ventas': 'sum', 'unidades': 'sum'},
    fill_value=0
)

# Calculamos métricas adicionales
# tabla_formateada['ventas_por_unidad'] = tabla_formateada['ventas'] / tabla_formateada['unidades']
# 
# # Reordenamos las columnas para mejor visualización
# tabla_formateada = tabla_formateada.reindex(columns=pd.MultiIndex.from_product(
#     [['ventas', 'unidades', 'ventas_por_unidad'], 
#      ['Norte', 'Sur', 'Este', 'Oeste']]
# ))
# 
# print("\nTabla de ventas con métricas adicionales:")
# print(tabla_formateada.round(2))


separator_line()

# Visualización de una tabla pivotante
import matplotlib.pyplot as plt

ventas_por_categoria = df_ventas.pivot_table(
    values='ventas',
    index='categoría',
    columns='región',
    aggfunc='sum',
    fill_value=0
)

# Creamos un gráfico de barras apiladas
ax = ventas_por_categoria.plot(
    kind='bar', 
    stacked=True,
    figsize=(10, 6),
    title='Ventas por categoría y región'
)

ax.set_ylabel('Ventas totales')
ax.legend(title='Región')

plt.tight_layout()
plt.show()


separator_line()

import pandas as pd
import numpy as np

# Creamos datos de ejemplo
data = {
    'producto': ['A', 'A', 'B', 'B', 'A', 'B'] * 5,
    'tienda': ['T1', 'T2', 'T1', 'T2', 'T3', 'T3'] * 5,
    'ventas': np.random.randint(100, 1000, 30),
    'descuento': np.random.uniform(0.05, 0.25, 30).round(2)
}

df = pd.DataFrame(data)

# Diferentes funciones de agregación
funciones_basicas = df.pivot_table(
    values='ventas',
    index='producto',
    columns='tienda',
    aggfunc=['sum', 'mean', 'count', 'min', 'max']
)

print("Tabla con múltiples funciones de agregación:")
print(funciones_basicas)


separator_line()
# Definimos funciones personalizadas
def rango(x):
    return x.max() - x.min()

def cv(x):  # Coeficiente de variación
    return x.std() / x.mean() if x.mean() != 0 else 0

# Aplicamos funciones personalizadas
tabla_personalizada = df.pivot_table(
    values='ventas',
    index='producto',
    columns='tienda',
    aggfunc=[np.sum, rango, cv]
)

print("\nTabla con funciones personalizadas:")
print(tabla_personalizada.round(2))

separator_line()

# Usando funciones lambda
tabla_lambda = df.pivot_table(
    values='ventas',
    index='producto',
    columns='tienda',
    aggfunc={
        'ventas': [
            ('Total', 'sum'),
            ('Promedio', 'mean'),
            ('Percentil 75', lambda x: np.percentile(x, 75))
        ]
    }
)

print("\nTabla con funciones lambda:")
print(tabla_lambda.round(2))

separator_line()

# Tabla pivotante con múltiples columnas de valores
multi_valores = df.pivot_table(
    values=['ventas', 'descuento'],
    index='producto',
    columns='tienda',
    aggfunc='mean'
)

print("\nTabla con múltiples valores:")
print(multi_valores.round(2))

separator_line()

# Diferentes funciones para diferentes columnas
multi_funciones = df.pivot_table(
    values=['ventas', 'descuento'],
    index='producto',
    columns='tienda',
    aggfunc={
        'ventas': 'sum',
        'descuento': 'mean'
    }
)

print("\nDiferentes funciones para diferentes valores:")
print(multi_funciones.round(2))


separator_line()
# Múltiples funciones para múltiples valores
tabla_compleja = df.pivot_table(
    values=['ventas', 'descuento'],
    index='producto',
    columns='tienda',
    aggfunc={
        'ventas': ['sum', 'mean', 'count'],
        'descuento': ['mean', 'min', 'max']
    }
)

print("\nMúltiples funciones para múltiples valores:")
print(tabla_compleja.round(2))

separator_line()

# Accediendo a datos específicos
ventas_totales_t1 = tabla_compleja[('ventas', 'sum', 'T1')]
descuento_promedio = tabla_compleja[('descuento', 'mean')]

print(f"\nVentas totales en T1 para producto A: {ventas_totales_t1['A']}")
print(f"Descuento promedio por tienda para producto B:")
print(descuento_promedio['B'])


separator_line()

# Accediendo a datos específicos
ventas_totales_t1 = tabla_compleja[('ventas', 'sum', 'T1')]
descuento_promedio = tabla_compleja[('descuento', 'mean')]

print(f"\nVentas totales en T1 para producto A: {ventas_totales_t1['A']}")
print(f"Descuento promedio por tienda para producto B:")
print(descuento_promedio['B'])

separator_line()

# Reorganizando la tabla para mejor visualización
tabla_reorganizada = tabla_compleja.stack(level=0)  # Apilamos el primer nivel de columnas

print("\nTabla reorganizada:")
print(tabla_reorganizada.round(2))


separator_line()

# Intercambiando niveles
tabla_intercambiada = tabla_compleja.swaplevel(0, 1, axis=1)
tabla_intercambiada = tabla_intercambiada.sort_index(axis=1)

print("\nTabla con niveles intercambiados:")
print(tabla_intercambiada.round(2))


separator_line()

# Datos más realistas para análisis de ventas
ventas_data = {
    'fecha': pd.date_range('2023-01-01', periods=100),
    'producto': np.random.choice(['Laptop', 'Tablet', 'Smartphone', 'Accesorios'], 100),
    'región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 100),
    'ventas': np.random.randint(200, 1500, 100),
    'unidades': np.random.randint(1, 10, 100),
    'costos': np.random.randint(100, 800, 100)
}

df_ventas = pd.DataFrame(ventas_data)

# Calculamos el margen
df_ventas['margen'] = df_ventas['ventas'] - df_ventas['costos']

# Análisis completo de ventas
analisis_ventas = df_ventas.pivot_table(
    values=['ventas', 'unidades', 'margen'],
    index='producto',
    columns='región',
    aggfunc={
        'ventas': ['sum', 'mean'],
        'unidades': 'sum',
        'margen': ['sum', lambda x: (x.sum() / df_ventas.loc[x.index, 'ventas'].sum() * 100).round(1)]
    },
    margins=True,
    margins_name='Total'
)

print("\nAnálisis completo de ventas:")
print(analisis_ventas)

separator_line()

# Agregamos trimestre a nuestros datos
df_ventas['trimestre'] = df_ventas['fecha'].dt.quarter

# Análisis temporal
analisis_temporal = df_ventas.pivot_table(
    values=['ventas', 'margen'],
    index='producto',
    columns='trimestre',
    aggfunc={
        'ventas': 'sum',
        'margen': [
            ('Total', 'sum'),
            ('% sobre ventas', lambda x: (x.sum() / df_ventas.loc[x.index, 'ventas'].sum() * 100).round(1))
        ]
    }
)

print("\nAnálisis temporal por trimestre:")
print(analisis_temporal)

separator_line()

# Preagregación de datos para mejorar rendimiento
df_preaggregated = df_ventas.groupby(['producto', 'región']).agg({
    'ventas': ['sum', 'mean'],
    'unidades': 'sum',
    'margen': 'sum'
}).reset_index()

# Ahora creamos la tabla pivotante con datos preagregados
tabla_optimizada = df_preaggregated.pivot_table(
    index='producto',
    columns='región',
    values=[('ventas', 'sum'), ('ventas', 'mean'), 
            ('unidades', 'sum'), ('margen', 'sum')]
)

print("\nTabla optimizada con datos preagregados:")
print(tabla_optimizada.round(2))


separator_line()

import pandas as pd
import numpy as np

# Creamos datos de ejemplo
data = {
    'categoría': ['Electrónica', 'Electrónica', 'Ropa', 'Ropa', 'Electrónica', 'Ropa'] * 5,
    'subcategoría': ['Móviles', 'Portátiles', 'Camisetas', 'Pantalones', 'Tablets', 'Abrigos'] * 5,
    'región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 30),
    'ventas': np.random.randint(1000, 5000, 30),
    'unidades': np.random.randint(10, 100, 30)
}

df = pd.DataFrame(data)

# Creamos una tabla pivotante con índices jerárquicos
tabla_jerarquica = df.pivot_table(
    values=['ventas', 'unidades'],
    index=['categoría', 'subcategoría'],
    columns='región',
    aggfunc='sum'
)

print("Tabla con índices jerárquicos:")
print(tabla_jerarquica)


separator_line()

# Explorando la estructura del índice
print("\nEstructura del índice de filas:")
print(tabla_jerarquica.index)

print("\nEstructura del índice de columnas:")
print(tabla_jerarquica.columns)

# Niveles individuales
print("\nNiveles del índice de filas:")
for i, nivel in enumerate(tabla_jerarquica.index.levels):
    print(f"Nivel {i}: {nivel.tolist()}")


separator_line()

# Acceso básico a datos
print("\nVentas de Electrónica en todas las regiones:")
print(tabla_jerarquica.loc['Electrónica'])

# Acceso a un valor específico
print("\nVentas de Móviles en la región Norte:")
print(tabla_jerarquica.loc[('Electrónica', 'Móviles'), ('ventas', 'Norte')])

# Selección por nivel
print("\nVentas de todas las subcategorías de Ropa:")
print(tabla_jerarquica.xs('Ropa', level='categoría'))

# Selección cruzada por nivel
print("\nVentas en la región Este para todas las categorías:")
print(tabla_jerarquica.xs('Este', level='región', axis=1))


separator_line()

# Intercambiando niveles en el índice de filas
tabla_reordenada = tabla_jerarquica.swaplevel(0, 1, axis=0)
print("\nTabla con niveles de filas intercambiados:")
print(tabla_reordenada.head())

# Ordenando el índice después de intercambiar niveles
tabla_ordenada = tabla_reordenada.sort_index(axis=0)
print("\nTabla con índice ordenado:")
print(tabla_ordenada.head())


separator_line()
# Apilando el nivel de región (convirtiendo columnas en filas)
tabla_apilada = tabla_jerarquica.stack(level='región')
print("\nTabla con región apilada (convertida a filas):")
print(tabla_apilada.head())

# Desapilando el nivel de subcategoría (convirtiendo filas en columnas)
tabla_desapilada = tabla_jerarquica.unstack(level='subcategoría')
print("\nTabla con subcategoría desapilada (convertida a columnas):")
print(tabla_desapilada.head())


separator_line()

# Agregación por nivel de categoría
ventas_por_categoria = tabla_jerarquica['ventas'].sum(axis=1, level='categoría')
print("\nTotal de ventas por categoría:")
print(ventas_por_categoria)

# Agregación por nivel de región
ventas_por_region = tabla_jerarquica.sum(axis=0, level='región')
print("\nTotal de ventas y unidades por región:")
print(ventas_por_region)


separator_line()

# Añadiendo un nuevo nivel al índice
df['temporada'] = np.random.choice(['Primavera', 'Verano', 'Otoño', 'Invierno'], 30)
tabla_tres_niveles = df.pivot_table(
    values='ventas',
    index=['categoría', 'subcategoría', 'temporada'],
    columns='región',
    aggfunc='sum'
)

print("\nTabla con tres niveles en el índice:")
print(tabla_tres_niveles.head())

# Eliminando un nivel mediante droplevel
tabla_simplificada = tabla_tres_niveles.droplevel('temporada', axis=0)
print("\nTabla con nivel eliminado:")
print(tabla_simplificada.head())


separator_line()

# Datos más completos para análisis
ventas_data = {
    'fecha': pd.date_range('2023-01-01', periods=200),
    'categoría': np.random.choice(['Electrónica', 'Ropa', 'Hogar'], 200),
    'subcategoría': np.random.choice(['Producto A', 'Producto B', 'Producto C', 'Producto D'], 200),
    'región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 200),
    'canal': np.random.choice(['Online', 'Tienda'], 200),
    'ventas': np.random.randint(500, 5000, 200),
    'unidades': np.random.randint(1, 50, 200)
}

df_ventas = pd.DataFrame(ventas_data)
df_ventas['trimestre'] = df_ventas['fecha'].dt.quarter

# Análisis multidimensional
analisis_completo = df_ventas.pivot_table(
    values=['ventas', 'unidades'],
    index=['categoría', 'subcategoría'],
    columns=['trimestre', 'canal'],
    aggfunc='sum',
    fill_value=0
)

print("\nAnálisis multidimensional de ventas:")
print(analisis_completo.head())

separator_line()

# Análisis específico: Ventas online por trimestre para productos de Electrónica
ventas_online_electronica = analisis_completo.xs(
    ('Electrónica'), 
    level='categoría'
).xs(
    'Online', 
    level='canal', 
    axis=1, 
    drop_level=False
)

print("\nVentas online por trimestre para productos de Electrónica:")
print(ventas_online_electronica)

# Comparación entre canales para un trimestre específico
comparacion_canales = analisis_completo.xs(
    1, 
    level='trimestre', 
    axis=1, 
    drop_level=False
)

print("\nComparación entre canales para el primer trimestre:")
print(comparacion_canales.head())


separator_line()

# Optimización: reconstruir índice para mejor rendimiento
tabla_optimizada = analisis_completo.copy()
tabla_optimizada.sort_index(inplace=True)  # Ordenar mejora el rendimiento de búsqueda

# Selección optimizada usando .loc con tuplas
resultado = tabla_optimizada.loc[('Electrónica', 'Producto A'), (slice(None), 'Online')]
print("\nSelección optimizada:")
print(resultado)


separator_line()

# Preparando datos para visualización
datos_viz = analisis_completo.sum(axis=1, level='categoría')
datos_viz = datos_viz.xs('ventas', axis=1, level=0)

# Código para visualización (representación textual)
print("\nDatos preparados para visualización:")
print(datos_viz)

# Ejemplo de cómo se vería el código para graficar
"""
import matplotlib.pyplot as plt

# Gráfico de barras para ventas por categoría y canal
ax = datos_viz.plot(
    kind='bar',
    figsize=(10, 6),
    title='Ventas por categoría y trimestre'
)
ax.set_ylabel('Ventas totales')
ax.set_xlabel('Categoría')
plt.tight_layout()
plt.show()
"""

