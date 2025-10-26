import pandas as pd
import numpy as np

def separator_line():
    print("\n" + "-"*50 + "\n")

# Creamos dos DataFrames con datos de ventas de diferentes trimestres
ventas_q1 = pd.DataFrame({
    'producto': ['A', 'B', 'C'],
    'ventas': [100, 200, 150]
})

ventas_q2 = pd.DataFrame({
    'producto': ['A', 'B', 'D'],
    'ventas': [120, 210, 190]
})

# Concatenamos verticalmente (por filas)
ventas_semestre = pd.concat([ventas_q1, ventas_q2], axis=0)
print("Concatenación vertical:")
print(ventas_semestre)

separator_line()

# Concatenación con índices reseteados
ventas_semestre2 = pd.concat([ventas_q1, ventas_q2], axis=0, ignore_index=True)
print("\nConcatenación con índices reseteados:")
print(ventas_semestre2)

separator_line()

# Datos de costos
costos_q1 = pd.DataFrame({
    'producto': ['A', 'B', 'C'],
    'costo': [50, 80, 70]
})

# Concatenación horizontal
datos_q1 = pd.concat([ventas_q1, costos_q1['costo']], axis=1)
print("\nConcatenación horizontal:")
print(datos_q1)

separator_line()

# Concatenación con etiquetas jerárquicas
etiquetado = pd.concat([ventas_q1, ventas_q2], keys=['Q1', 'Q2'])
print("\nConcatenación con etiquetas jerárquicas:")
print(etiquetado)

separator_line()

# Datos de productos
productos = pd.DataFrame({
    'producto_id': ['A', 'B', 'C', 'D'],
    'nombre': ['Laptop', 'Monitor', 'Teclado', 'Mouse'],
    'categoria': ['Computadoras', 'Periféricos', 'Periféricos', 'Periféricos']
})

# Datos de ventas
ventas = pd.DataFrame({
    'producto_id': ['A', 'B', 'B', 'C'],
    'tienda': ['Central', 'Norte', 'Sur', 'Central'],
    'unidades': [5, 3, 2, 7]
})

# Combinación inner (solo registros que coinciden en ambos DataFrames)
ventas_productos = pd.merge(ventas, productos, on='producto_id')
print("Merge inner:")
print(ventas_productos)
separator_line()

# Left join (mantiene todas las filas del DataFrame izquierdo)
left_join = pd.merge(ventas, productos, on='producto_id', how='left')
print("\nLeft join:")
print(left_join)

# Right join (mantiene todas las filas del DataFrame derecho)
right_join = pd.merge(ventas, productos, on='producto_id', how='right')
print("\nRight join:")
print(right_join)

# Outer join (mantiene todas las filas de ambos DataFrames)
outer_join = pd.merge(ventas, productos, on='producto_id', how='outer')
print("\nOuter join:")
print(outer_join)
separator_line()

# DataFrame con nombre de columna diferente
inventario = pd.DataFrame({
    'codigo': ['A', 'B', 'C', 'E'],
    'stock': [15, 10, 20, 5]
})

# Merge con nombres de columnas diferentes
merge_diff_cols = pd.merge(
    ventas, 
    inventario, 
    left_on='producto_id', 
    right_on='codigo'
)
print("\nMerge con nombres de columnas diferentes:")
print(merge_diff_cols)
separator_line()

# Configuramos índices
productos_indexed = productos.set_index('producto_id')
ventas_indexed = ventas.set_index('producto_id')

# Merge por índice
merge_indexed = pd.merge(
    ventas_indexed, 
    productos_indexed, 
    left_index=True, 
    right_index=True
)
print("\nMerge por índice:")
print(merge_indexed)
separator_line()

# Join (equivalente a merge con left_index=True, right_index=True)
join_result = ventas_indexed.join(productos_indexed)
print("Join por índice:")
print(join_result)
separator_line()

# Join con diferentes tipos
join_left = ventas_indexed.join(inventario.set_index('codigo'), how='left')
print("\nLeft join con método join():")
print(join_left)
separator_line()

# Ejemplo práctico: Análisis de ventas completo
# Combinamos ventas con productos e inventario
analisis_completo = (ventas
    .merge(productos, left_on='producto_id', right_on='producto_id')
    .merge(inventario, left_on='producto_id', right_on='codigo')
)

# Calculamos ratio de ventas/stock
analisis_completo['ratio_venta_stock'] = analisis_completo['unidades'] / analisis_completo['stock']
print("\nAnálisis completo de ventas:")
print(analisis_completo[['producto_id', 'nombre', 'tienda', 'unidades', 'stock', 'ratio_venta_stock']])
separator_line()

# DataFrames con columnas de mismo nombre
df1 = pd.DataFrame({'clave': ['A', 'B', 'C'], 'valor': [1, 2, 3], 'fecha': ['2023-01-01', '2023-01-02', '2023-01-03']})
df2 = pd.DataFrame({'clave': ['A', 'B', 'D'], 'valor': [10, 20, 40], 'región': ['Norte', 'Sur', 'Este']})

# Merge con sufijos personalizados
merge_sufijos = pd.merge(df1, df2, on='clave', suffixes=('_df1', '_df2'))
print("Merge con sufijos personalizados:")
print(merge_sufijos)
separator_line()

# Merge con indicador
merge_indicador = pd.merge(
    df1, df2, on='clave', how='outer', indicator=True
)
print("\nMerge con indicador de origen:")
print(merge_indicador)
separator_line()

# Ejemplo de merge_asof para datos temporales (útil en series financieras)
precios = pd.DataFrame({
    'fecha': pd.date_range('2023-01-01', periods=5),
    'precio': [100, 102, 99, 101, 103]
})

compras = pd.DataFrame({
    'fecha': pd.date_range('2023-01-01', periods=3, freq='2D'),
    'volumen': [10, 15, 20]
})

# Ordenamos los datos
precios = precios.sort_values('fecha')
compras = compras.sort_values('fecha')

# Merge asof (combina con el valor previo más cercano)
from pandas import merge_asof
resultado_asof = merge_asof(compras, precios, on='fecha')
print("\nMerge asof para datos temporales:")
print(resultado_asof)
separator_line()

import pandas as pd
import numpy as np

# Creamos un DataFrame en formato ancho (ventas por producto y trimestre)
datos_ancho = pd.DataFrame({
    'producto': ['Laptop', 'Tablet', 'Móvil', 'Auriculares'],
    'Q1': [350, 270, 410, 170],
    'Q2': [390, 280, 430, 160],
    'Q3': [320, 290, 380, 190],
    'Q4': [400, 310, 450, 210]
})

print("Datos en formato ancho:")
print(datos_ancho)
separator_line()

# Transformamos de ancho a largo con melt()
datos_largo = pd.melt(
    datos_ancho,
    id_vars=['producto'],     # Columnas que mantienen su estructura
    value_vars=['Q1', 'Q2', 'Q3', 'Q4'],  # Columnas a "derretir"
    var_name='trimestre',     # Nombre para la columna de variables
    value_name='ventas'       # Nombre para la columna de valores
)

print("\nDatos transformados a formato largo con melt():")
print(datos_largo)
separator_line()

# Melt con valores nulos explícitos
datos_con_nulos = datos_ancho.copy()
datos_con_nulos.loc[0, 'Q3'] = np.nan

datos_largo_con_nulos = pd.melt(
    datos_con_nulos,
    id_vars=['producto'],
    value_vars=['Q1', 'Q2', 'Q3', 'Q4'],
    var_name='trimestre',
    value_name='ventas',
    ignore_index=False  # Mantener el índice original
)

print("\nMelt con valores nulos:")
print(datos_largo_con_nulos)
separator_line()
'''La función pivot() realiza la operación inversa a melt(), convirtiendo datos de formato largo a ancho. Es como crear una tabla dinámica en Excel.'''

# Primero creamos datos en formato largo
datos_empleados = pd.DataFrame({
    'empleado': ['Ana', 'Ana', 'Ana', 'Carlos', 'Carlos', 'Carlos'],
    'año': [2021, 2022, 2023, 2021, 2022, 2023],
    'salario': [50000, 52000, 54000, 45000, 47000, 49000]
})

print("\nDatos originales en formato largo:")
print(datos_empleados)

# Transformamos a formato ancho con pivot()
tabla_pivote = datos_empleados.pivot(
    index='empleado',    # Filas de la tabla
    columns='año',       # Columnas de la tabla
    values='salario'     # Valores a mostrar
)

print("\nDatos transformados a formato ancho con pivot():")
print(tabla_pivote)
separator_line()

# Datos con valores duplicados
ventas_regionales = pd.DataFrame({
    'región': ['Norte', 'Norte', 'Sur', 'Sur', 'Norte', 'Sur'],
    'producto': ['A', 'B', 'A', 'B', 'A', 'B'],
    'ventas': [100, 150, 200, 250, 120, 230]
})

print("\nDatos con valores duplicados:")
print(ventas_regionales)

# Creamos una tabla pivote con agregación
tabla_agregada = pd.pivot_table(
    ventas_regionales,
    index='región',
    columns='producto',
    values='ventas',
    aggfunc='sum'  # Podemos usar 'mean', 'count', 'max', etc.
)

print("\nTabla pivote con agregación (suma):")
print(tabla_agregada)
separator_line()

# Tabla pivote con múltiples funciones de agregación
tabla_multi_agg = pd.pivot_table(
    ventas_regionales,
    index='región',
    columns='producto',
    values='ventas',
    aggfunc=['sum', 'mean', 'count']
)

print("\nTabla pivote con múltiples agregaciones:")
print(tabla_multi_agg)

# Tabla pivote con valores de relleno y totales
tabla_completa = pd.pivot_table(
    ventas_regionales,
    index='región',
    columns='producto',
    values='ventas',
    aggfunc='sum',
    fill_value=0,       # Valor para celdas vacías
    margins=True,       # Añadir totales
    margins_name='Total'  # Nombre para la fila/columna de totales
)

print("\nTabla pivote con totales:")
print(tabla_completa)

separator_line()

# Creamos un DataFrame con índice multinivel
ventas_multi = pd.DataFrame({
    ('2022', 'Q1'): [100, 200],
    ('2022', 'Q2'): [110, 210],
    ('2023', 'Q1'): [120, 220],
    ('2023', 'Q2'): [130, 230]
}, index=['Producto A', 'Producto B'])

# Convertimos el encabezado a multinivel
ventas_multi.columns = pd.MultiIndex.from_tuples(ventas_multi.columns, names=['Año', 'Trimestre'])

print("\nDataFrame con índice multinivel:")
print(ventas_multi)
separator_line()

# Apilamos el nivel 'Trimestre'
ventas_stacked = ventas_multi.stack(level='Trimestre')
print("\nDataFrame después de stack() en nivel 'Trimestre':")
print(ventas_stacked)
separator_line()

# Desapilamos el nivel 0 del índice (Producto)
ventas_unstacked = ventas_stacked.unstack(level=0)
print("\nDataFrame después de unstack() en nivel 0:")
print(ventas_unstacked)
separator_line()

# Datos de ventas por región, producto y mes
ventas_detalladas = pd.DataFrame({
    'fecha': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'región': ['Norte', 'Sur', 'Este', 'Oeste'] * 3,
    'producto': ['A', 'B', 'A', 'B'] * 3,
    'unidades': np.random.randint(50, 150, 12),
    'precio_unitario': np.random.uniform(10, 30, 12).round(2)
})

# Calculamos el valor total
ventas_detalladas['valor_total'] = ventas_detalladas['unidades'] * ventas_detalladas['precio_unitario']

# Extraemos mes y trimestre
ventas_detalladas['mes'] = ventas_detalladas['fecha'].dt.month_name()
ventas_detalladas['trimestre'] = 'Q' + ventas_detalladas['fecha'].dt.quarter.astype(str)

print("\nDatos de ventas detalladas:")
print(ventas_detalladas.head())

# Análisis por trimestre y región
analisis_trimestral = pd.pivot_table(
    ventas_detalladas,
    index=['región', 'producto'],
    columns='trimestre',
    values='valor_total',
    aggfunc='sum'
)

print("\nAnálisis trimestral por región y producto:")
print(analisis_trimestral)

# Reorganizamos para comparar productos dentro de cada región
comparativa_productos = analisis_trimestral.unstack(level='producto')
print("\nComparativa de productos por región:")
print(comparativa_productos)
separator_line()

# Caso 1: Convertir datos de encuestas a formato adecuado para visualización
encuestas = pd.DataFrame({
    'encuestado': ['P1', 'P2', 'P3', 'P4'],
    'pregunta1': [5, 4, 3, 5],
    'pregunta2': [4, 3, 5, 4],
    'pregunta3': [3, 5, 4, 3]
})

# Transformamos a formato largo para visualización
encuestas_viz = pd.melt(
    encuestas, 
    id_vars=['encuestado'],
    var_name='pregunta', 
    value_name='puntuación'
)

print("\nDatos de encuesta transformados para visualización:")
print(encuestas_viz.head())

# Caso 2: Análisis de series temporales con múltiples variables
datos_sensor = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=24, freq='H'),
    'temperatura': np.random.normal(22, 3, 24).round(1),
    'humedad': np.random.normal(60, 10, 24).round(1),
    'presión': np.random.normal(1013, 5, 24).round(1)
})

# Transformamos para análisis de correlación entre variables
datos_sensor_largo = pd.melt(
    datos_sensor,
    id_vars=['timestamp'],
    var_name='medida',
    value_name='valor'
)

print("\nDatos de sensores transformados para análisis:")
print(datos_sensor_largo.head())

# Volvemos a formato ancho pero con timestamp como índice
datos_sensor_ancho = datos_sensor_largo.pivot(
    index='timestamp',
    columns='medida',
    values='valor'
)

print("\nDatos de sensores reorganizados con timestamp como índice:")
print(datos_sensor_ancho.head())
separator_line()

# Ejemplo de optimización para datos grandes
# Supongamos que tenemos un DataFrame grande
# Podemos usar copy=False para evitar duplicar datos en memoria

# Transformación eficiente
datos_largo_eficiente = pd.melt(
    datos_ancho,
    id_vars=['producto'],
    value_vars=['Q1', 'Q2', 'Q3', 'Q4'],
    var_name='trimestre',
    value_name='ventas'
    # copy=False  # Evita copiar datos cuando es posible
)

separator_line()

# Creamos un DataFrame de ventas
ventas = pd.DataFrame({
    'fecha': pd.date_range('2023-01-01', periods=10),
    'vendedor': ['Ana', 'Carlos', 'Ana', 'Beatriz', 'Carlos', 
                'Ana', 'Beatriz', 'Carlos', 'Ana', 'Beatriz'],
    'producto': ['Laptop', 'Tablet', 'Monitor', 'Laptop', 'Smartphone',
                'Tablet', 'Monitor', 'Laptop', 'Smartphone', 'Tablet'],
    'unidades': [3, 5, 2, 4, 7, 2, 3, 5, 1, 6],
    'precio': [1200, 500, 300, 1200, 800, 500, 300, 1200, 800, 500]
})

# Calculamos el valor total de cada venta
ventas['valor_total'] = ventas['unidades'] * ventas['precio']

print("DataFrame de ventas:")
print(ventas)

# Agrupamos por vendedor y calculamos la suma de unidades vendidas
ventas_por_vendedor = ventas.groupby('vendedor')['unidades'].sum()
print("\nTotal de unidades vendidas por vendedor:")
print(ventas_por_vendedor)

separator_line()

# Estadísticas básicas por vendedor
estadisticas_vendedor = ventas.groupby('vendedor')['valor_total'].agg(['sum', 'mean', 'count'])
print("\nEstadísticas de ventas por vendedor:")
print(estadisticas_vendedor)

# Valor mínimo, máximo y total de ventas por producto
estadisticas_producto = ventas.groupby('producto').agg({
    'unidades': 'sum',
    'valor_total': ['min', 'max', 'sum']
})
print("\nEstadísticas por producto:")
print(estadisticas_producto)
separator_line()

# Agrupamos por vendedor y producto
ventas_detalladas = ventas.groupby(['vendedor', 'producto'])['valor_total'].sum()
print("\nVentas por vendedor y producto:")
print(ventas_detalladas)

# Convertimos el resultado a DataFrame para mejor visualización
ventas_matriz = ventas_detalladas.unstack(fill_value=0)
print("\nMatriz de ventas (vendedor × producto):")
print(ventas_matriz)
separator_line()

# Iteramos sobre los grupos por vendedor
print("\nAnálisis detallado por vendedor:")
for nombre, grupo in ventas.groupby('vendedor'):
    print(f"\nVendedor: {nombre}")
    print(f"Total de ventas: ${grupo['valor_total'].sum():,.2f}")
    print(f"Producto más vendido: {grupo.groupby('producto')['unidades'].sum().idxmax()}")
    print(f"Número de transacciones: {len(grupo)}")

separator_line()
# Aplicamos diferentes funciones a distintas columnas
resumen_completo = ventas.groupby('vendedor').agg({
    'unidades': ['sum', 'mean'],
    'valor_total': ['sum', 'mean', 'max'],
    'fecha': ['min', 'max']  # Primera y última venta
})

print("\nResumen completo por vendedor:")
print(resumen_completo)
separator_line()

# Definimos una función personalizada para calcular el ratio de valor/unidades
def valor_por_unidad(x):
    return x.sum() / len(x)

# Función para calcular el porcentaje del total
def porcentaje_del_total(x):
    return x.sum() / ventas['valor_total'].sum() * 100

# Aplicamos funciones personalizadas
metricas_avanzadas = ventas.groupby('vendedor')['valor_total'].agg([
    'sum',
    ('promedio_por_venta', valor_por_unidad),
    ('porcentaje_total', porcentaje_del_total)
])

print("\nMétricas avanzadas por vendedor:")
print(metricas_avanzadas)
separator_line()

# Calculamos el porcentaje que representa cada venta dentro de su grupo (vendedor)
ventas['porcentaje_vendedor'] = ventas.groupby('vendedor')['valor_total'].transform(
    lambda x: x / x.sum() * 100
)

# Añadimos el ranking de cada venta dentro de su grupo
ventas['ranking_vendedor'] = ventas.groupby('vendedor')['valor_total'].transform(
    lambda x: x.rank(ascending=False)
)

print("\nVentas con métricas relativas al grupo:")
print(ventas[['vendedor', 'producto', 'valor_total', 'porcentaje_vendedor', 'ranking_vendedor']])
separator_line()


# Filtramos para mostrar solo vendedores con más de 10 unidades vendidas
vendedores_top = ventas.groupby('vendedor').filter(lambda x: x['unidades'].sum() > 10)
print("\nVendedores con más de 10 unidades vendidas:")
print(vendedores_top[['vendedor', 'unidades', 'valor_total']])

# Filtramos para mostrar solo productos con valor total superior a 5000
productos_premium = ventas.groupby('producto').filter(lambda x: x['valor_total'].sum() > 5000)
print("\nProductos con valor total superior a 5000:")
print(productos_premium[['producto', 'unidades', 'valor_total']])
separator_line()

# Añadimos más datos temporales para el ejemplo
ventas_temporales = pd.DataFrame({
    'fecha': pd.date_range('2023-01-01', periods=100, freq='D'),
    'ventas': np.random.randint(100, 1000, 100)
})

# Agrupamos por mes
ventas_mensuales = ventas_temporales.groupby(ventas_temporales['fecha'].dt.month)['ventas'].sum()
print("\nVentas mensuales:")
print(ventas_mensuales)

# Agrupamos por semana del año
ventas_semanales = ventas_temporales.groupby(ventas_temporales['fecha'].dt.isocalendar().week)['ventas'].agg(['sum', 'mean'])
print("\nVentas semanales:")
print(ventas_semanales)

# Agrupamos por día de la semana
ventas_por_dia = ventas_temporales.groupby(ventas_temporales['fecha'].dt.day_name())['ventas'].mean()
print("\nVentas promedio por día de la semana:")
print(ventas_por_dia)
separator_line()

# Configuramos la fecha como índice
ventas_temporales = ventas_temporales.set_index('fecha')

# Remuestreamos a nivel mensual
ventas_mes = ventas_temporales.resample('M').sum()
print("\nVentas mensuales con resample():")
print(ventas_mes)

# Remuestreamos a nivel semanal
ventas_semana = ventas_temporales.resample('W').agg(['sum', 'mean', 'max'])
print("\nEstadísticas semanales:")
print(ventas_semana.head())
separator_line()

# Creamos datos de acciones para dos empresas
acciones = pd.DataFrame({
    'fecha': pd.date_range('2023-01-01', periods=20),
    'empresa': ['TechCorp', 'EcoSolutions'] * 10,
    'precio': np.random.normal(100, 10, 20).round(2)
})

# Configuramos índice múltiple
acciones = acciones.set_index(['empresa', 'fecha'])

# Calculamos la media móvil de 3 días para cada empresa
media_movil = acciones.groupby(level='empresa')['precio'].rolling(window=3).mean()
print("\nMedia móvil de precios por empresa:")
print(media_movil)

# Resetamos el índice para mejor visualización
#media_movil = media_movil.reset_index()
#print("\nMedia móvil con índice reseteado:")
#print(media_movil)
separator_line()

# Agrupación con categorías para mejorar rendimiento
ventas['vendedor_cat'] = ventas['vendedor'].astype('category')
ventas['producto_cat'] = ventas['producto'].astype('category')

# La agrupación con tipos categóricos es más eficiente
resultado_optimizado = ventas.groupby(['vendedor_cat', 'producto_cat'])['valor_total'].sum()

# Para cálculos muy intensivos, podemos usar numba (si está instalada)
# from numba import jit
# @jit(nopython=True)
# def suma_rapida(x):
#     return x.sum()
# 
# ventas.groupby('vendedor')['valor_total'].apply(suma_rapida)

separator_line()

# Caso 1: Análisis de rendimiento de ventas
# Calculamos métricas de rendimiento por vendedor y producto

rendimiento = ventas.groupby(['vendedor', 'producto']).agg({
    'unidades': 'sum',
    'valor_total': 'sum'
}).reset_index()

# Añadimos métricas derivadas
rendimiento['valor_por_unidad'] = rendimiento['valor_total'] / rendimiento['unidades']

# Calculamos el ranking de productos por vendedor
rendimiento['ranking'] = rendimiento.groupby('vendedor')['valor_total'].rank(ascending=False)

print("\nAnálisis de rendimiento:")
print(rendimiento)

# Caso 2: Detección de anomalías
# Identificamos ventas que se desvían significativamente del promedio del grupo

# Calculamos estadísticas por producto
stats_producto = ventas.groupby('producto')['valor_total'].agg(['mean', 'std']).reset_index()
stats_producto.columns = ['producto', 'media_valor', 'desv_valor']

# Unimos con el DataFrame original
ventas_con_stats = pd.merge(ventas, stats_producto, on='producto')

# Calculamos el z-score para cada venta
ventas_con_stats['z_score'] = (ventas_con_stats['valor_total'] - ventas_con_stats['media_valor']) / ventas_con_stats['desv_valor']

# Identificamos anomalías (z-score > 2 o < -2)
anomalias = ventas_con_stats[abs(ventas_con_stats['z_score']) > 1.5]
print("\nVentas anómalas detectadas:")
print(anomalias[['vendedor', 'producto', 'valor_total', 'z_score']])
separator_line()

'''El método apply() es probablemente el más versátil y permite ejecutar una función a lo largo de un eje específico de 
un DataFrame o Series. Puede trabajar tanto con filas como con columnas.'''

# Creamos un DataFrame de ejemplo
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

print("DataFrame original:")
print(df)

separator_line()

# Calculamos estadísticas básicas para cada columna
estadisticas = df.apply(lambda x: pd.Series({
    'suma': x.sum(),
    'media': x.mean(),
    'max': x.max(),
    'min': x.min()
}))

print("\nEstadísticas por columna:")
print(estadisticas)
separator_line()

# Calculamos la suma de cada fila
df['suma_fila'] = df.apply(lambda x: x.sum(), axis=1)

# Calculamos el valor máximo de cada fila
df['max_fila'] = df.apply(lambda x: x.max(), axis=1)

print("\nDataFrame con cálculos por fila:")
print(df)

separator_line()

# Función para normalizar valores en una columna
def normalizar(columna):
    return (columna - columna.min()) / (columna.max() - columna.min())

# Aplicamos la normalización a cada columna numérica
df_normalizado = df[['A', 'B', 'C']].apply(normalizar)

print("\nDataFrame con columnas normalizadas:")
print(df_normalizado)

# Función para clasificar valores en una fila
def clasificar_fila(fila):
    if fila.sum() > 500:
        return "Alto"
    elif fila.sum() > 300:
        return "Medio"
    else:
        return "Bajo"

# Aplicamos la clasificación a cada fila
df['categoria'] = df[['A', 'B', 'C']].apply(clasificar_fila, axis=1)

print("\nDataFrame con categorización por fila:")
print(df)
separator_line()

# Función que devuelve un diccionario con múltiples cálculos
def analizar_columna(col):
    return {
        'media': col.mean(),
        'desviacion': col.std(),
        'mediana': col.median(),
        'rango': col.max() - col.min()
    }

# Aplicamos la función a cada columna
analisis = df[['A', 'B', 'C']].apply(analizar_columna)

# Convertimos el resultado a un DataFrame para mejor visualización
analisis_df = pd.DataFrame(analisis)
print("\nAnálisis detallado por columna:")
print(analisis_df)
separator_line()

# Creamos una Series de ejemplo
calificaciones = pd.Series([85, 92, 63, 78, 95])
print("\nCalificaciones originales:")
print(calificaciones)

# Mapeamos valores numéricos a categorías
def asignar_letra(nota):
    if nota >= 90:
        return 'A'
    elif nota >= 80:
        return 'B'
    elif nota >= 70:
        return 'C'
    elif nota >= 60:
        return 'D'
    else:
        return 'F'

calificaciones_letras = calificaciones.map(asignar_letra)
print("\nCalificaciones en letras:")
print(calificaciones_letras)


separator_line()

# Creamos una Series con códigos de productos
productos = pd.Series(['P001', 'P002', 'P003', 'P001', 'P004'])
print("\nCódigos de productos:")
print(productos)

# Diccionario de mapeo código -> nombre
nombres_productos = {
    'P001': 'Laptop',
    'P002': 'Monitor',
    'P003': 'Teclado',
    'P004': 'Mouse'
}

# Mapeamos códigos a nombres
nombres = productos.map(nombres_productos)
print("\nNombres de productos:")
print(nombres)
separator_line()

# Series con precios
precios = pd.Series([120.50, 99.99, 45.75, 199.99, 25.50])
print("\nPrecios originales:")
print(precios)

# Aplicamos un descuento del 10%
precios_descuento = precios.map(lambda x: round(x * 0.9, 2))
print("\nPrecios con 10% de descuento:")
print(precios_descuento)
separator_line()

# Creamos un DataFrame con valores mixtos
df_mixto = pd.DataFrame({
    'A': [1, -2, 3, -4, 5],
    'B': [0.1, 0.2, -0.3, 0.4, -0.5],
    'C': ['10%', '20%', '30%', '40%', '50%']
})

print("\nDataFrame con valores mixtos:")
print(df_mixto)

# Aplicamos valor absoluto a todos los elementos numéricos
# y dejamos las cadenas sin cambios
def valor_abs(x):
    try:
        return abs(float(x.strip('%')) if isinstance(x, str) and '%' in x else x)
    except:
        return x

df_abs = df_mixto.applymap(valor_abs)
print("\nDataFrame con valores absolutos:")
print(df_abs)
separator_line()

# DataFrame con valores numéricos
df_numerico = pd.DataFrame({
    'A': [1.23456, 2.34567, 3.45678],
    'B': [10.1234, 20.2345, 30.3456],
    'C': [100.123, 200.234, 300.345]
})

print("\nDataFrame numérico original:")
print(df_numerico)

# Formateamos todos los valores a 2 decimales
df_formateado = df_numerico.applymap(lambda x: f"{x:.2f}")
print("\nDataFrame con valores formateados:")
print(df_formateado)
separator_line()

# DataFrame de ejemplo para comparación
datos = pd.DataFrame({
    'nombre': ['Ana', 'Carlos', 'Beatriz', 'David', 'Elena'],
    'edad': [28, 35, 42, 31, 25],
    'salario': [45000, 55000, 62000, 48000, 42000],
    'departamento': ['IT', 'Marketing', 'Finanzas', 'IT', 'Marketing']
})

print("\nDataFrame de empleados:")
print(datos)

# 1. Usando map() en una Series
# Categorizar departamentos
dept_categoria = {
    'IT': 'Tecnología',
    'Marketing': 'Comercial',
    'Finanzas': 'Administración'
}
datos['area'] = datos['departamento'].map(dept_categoria)

# 2. Usando apply() en columnas
# Calcular impuestos basados en salario
def calcular_impuesto(salario):
    if salario < 45000:
        return salario * 0.15
    elif salario < 60000:
        return salario * 0.20
    else:
        return salario * 0.25

datos['impuesto'] = datos['salario'].apply(calcular_impuesto)

# 3. Usando apply() en filas
# Calcular salario neto
datos['salario_neto'] = datos.apply(
    lambda x: x['salario'] - x['impuesto'], axis=1
)

# 4. Usando applymap() en un subconjunto de columnas
# Formatear valores numéricos
datos[['edad', 'salario', 'impuesto', 'salario_neto']] = datos[
    ['edad', 'salario', 'impuesto', 'salario_neto']
].applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

print("\nDataFrame con transformaciones aplicadas:")
print(datos)

separator_line()

# Creamos un DataFrame grande para pruebas
df_grande = pd.DataFrame(np.random.randn(100000, 5), columns=list('ABCDE'))

# Vectorización vs. apply
# La vectorización es generalmente más rápida
# %time 
resultado_vectorizado = np.sqrt(df_grande**2)  # Operación vectorizada

# %time 
resultado_apply = df_grande.apply(lambda x: np.sqrt(x**2))  # Con apply

# Para operaciones complejas, podemos usar numba para acelerar
# from numba import jit
# @jit(nopython=True)
# def operacion_rapida(x):
#     return np.sqrt(x**2)
# 
# %time resultado_numba = df_grande.apply(operacion_rapida)

separator_line()

# Caso 1: Procesamiento de texto en un conjunto de datos
datos_texto = pd.DataFrame({
    'id': range(1, 6),
    'texto': [
        "El producto es excelente, muy satisfecho.",
        "No funciona correctamente, decepcionado.",
        "Buena relación calidad-precio, recomendable.",
        "Llegó con retraso pero el producto es bueno.",
        "Pésima atención al cliente, no lo recomiendo."
    ]
})

# Extraemos métricas del texto
def analizar_texto(texto):
    palabras = texto.lower().split()
    return {
        'num_palabras': len(palabras),
        'longitud_media': sum(len(p) for p in palabras) / len(palabras) if palabras else 0,
        'sentimiento': 'positivo' if any(p in texto.lower() for p in ['excelente', 'buena', 'satisfecho', 'recomendable']) 
                       else 'negativo' if any(p in texto.lower() for p in ['no', 'pésima', 'decepcionado']) 
                       else 'neutro'
    }

# Aplicamos el análisis a cada texto
analisis_texto = datos_texto['texto'].apply(analizar_texto)
resultados_texto = pd.DataFrame(analisis_texto.tolist())
datos_texto = pd.concat([datos_texto, resultados_texto], axis=1)

print("\nAnálisis de textos:")
print(datos_texto)

# Caso 2: Transformación de datos financieros
datos_financieros = pd.DataFrame({
    'fecha': pd.date_range('2023-01-01', periods=5),
    'ingresos': [12500, 9800, 15200, 11300, 18500],
    'gastos': [8700, 7600, 9200, 8100, 10300],
    'moneda': ['USD', 'EUR', 'USD', 'GBP', 'USD']
})

# Tasas de cambio a USD
tasas_cambio = {'USD': 1.0, 'EUR': 1.1, 'GBP': 1.3}

# Convertimos todos los valores monetarios a USD
def convertir_a_usd(fila):
    tasa = tasas_cambio[fila['moneda']]
    return pd.Series({
        'ingresos_usd': fila['ingresos'] * tasa,
        'gastos_usd': fila['gastos'] * tasa,
        'beneficio_usd': (fila['ingresos'] - fila['gastos']) * tasa
    })

datos_usd = datos_financieros.apply(convertir_a_usd, axis=1)
datos_financieros = pd.concat([datos_financieros, datos_usd], axis=1)

print("\nDatos financieros convertidos a USD:")
print(datos_financieros)

separator_line()

# Creamos datos de ventas
ventas_tiendas = pd.DataFrame({
    'tienda': ['T1', 'T2', 'T3', 'T1', 'T2'],
    'producto': ['A', 'B', 'A', 'C', 'A'],
    'unidades': [10, 15, 8, 12, 20],
    'precio_unitario': [100, 150, 100, 80, 100]
})

# Transformación encadenada
resultado = (ventas_tiendas
    # Calculamos el valor total
    .assign(valor_total=lambda df: df['unidades'] * df['precio_unitario'])
    # Agrupamos por tienda
    .groupby('tienda')
    # Aplicamos múltiples cálculos
    .apply(lambda g: pd.Series({
        'total_ventas': g['valor_total'].sum(),
        'productos_diferentes': g['producto'].nunique(),
        'ticket_promedio': g['valor_total'].mean(),
        'mejor_producto': g.loc[g['valor_total'].idxmax(), 'producto']
    }))
    # Ordenamos por total de ventas
    .sort_values('total_ventas', ascending=False)
)

print("\nAnálisis de ventas por tienda (métodos encadenados):")
print(resultado)
separator_line()

# Creamos un DataFrame con valores numéricos
df_cientifico = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])
print("\nDataFrame para cálculos científicos:")
print(df_cientifico)

# Aplicamos funciones de NumPy a columnas
transformaciones = df_cientifico.apply({
    'A': np.exp,       # Función exponencial
    'B': np.sin,       # Seno
    'C': lambda x: np.sqrt(x**2)  # Valor absoluto mediante raíz cuadrada
})

print("\nColumnas transformadas con funciones de NumPy:")
print(transformaciones)

# Aplicamos estadísticas avanzadas a filas
estadisticas_filas = df_cientifico.apply(
    lambda x: pd.Series({
        'media': x.mean(),
        'desviacion': x.std(),
        'kurtosis': x.kurtosis(),
        'skewness': x.skew()
    }), 
    axis=1
)

print("\nEstadísticas avanzadas por fila:")
print(estadisticas_filas)
separator_line()
