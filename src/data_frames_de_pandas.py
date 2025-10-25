'''
'''
import pandas as pd
import numpy as np

def print_separator():
    '''Imprime una línea separadora'''
    print("-" * 40)


# Creamos un DataFrame simple
datos = {
    'nombre': ['Ana', 'Carlos', 'Lucía', 'Miguel'],
    'edad': [28, 32, 25, 41],
    'ciudad': ['Madrid', 'Barcelona', 'Sevilla', 'Valencia'],
    'activo': [True, True, False, True]
}

df = pd.DataFrame(datos)
print(df)

print_separator()

# Creando un DataFrame con índice personalizado
df = pd.DataFrame(datos, index=['p1', 'p2', 'p3', 'p4'])
print(df)
print_separator()

# Creando un DataFrame con índice jerárquico
idx = pd.MultiIndex.from_tuples([
    ('España', 'Madrid'),
    ('España', 'Barcelona'),
    ('España', 'Sevilla'),
    ('España', 'Valencia')
], names=['país', 'ciudad'])

df = pd.DataFrame({
    'nombre': ['Ana', 'Carlos', 'Lucía', 'Miguel'],
    'edad': [28, 32, 25, 41]
}, index=idx)

print(df)
print_separator()

# Explorando las columnas de un DataFrame
print(df.columns)  # Muestra los nombres de las columnas
print(df.dtypes)   # Muestra los tipos de datos de cada columna

print_separator()

# Diferentes formas de acceder a una columna
print(df['nombre'])     # Acceso tipo diccionario
print(df.nombre)        # Acceso tipo atributo (no funciona con nombres con espacios)
print_separator()

# DataFrame con diferentes tipos de datos
df_tipos = pd.DataFrame({
    'enteros': [1, 2, 3, 4],
    'flotantes': [1.1, 2.2, 3.3, 4.4],
    'cadenas': ['a', 'b', 'c', 'd'],
    'booleanos': [True, False, True, False],
    'fechas': pd.date_range('2023-01-01', periods=4)
})

print(df_tipos.dtypes)
print_separator()

# Usando los tipos de datos mejorados de Pandas 2
df_mejorado = pd.DataFrame({
    'enteros': pd.Series([1, 2, None, 4], dtype='Int64'),
    'texto': pd.Series(['a', 'b', 'c', None], dtype='string'),
    'categoría': pd.Series(['alto', 'medio', 'alto', 'bajo'], dtype='category'),
    'fecha': pd.Series(pd.date_range('2023-01-01', periods=3).tolist() + [None], 
                      dtype='datetime64[ns]')
})

print(df_mejorado.dtypes)
print(df_mejorado)
print_separator()

# Creamos un DataFrame más complejo para análisis
df_ventas = pd.DataFrame({
    'producto': ['A', 'B', 'C', 'D', 'E'] * 4,
    'región': ['Norte', 'Sur', 'Este', 'Oeste'] * 5,
    'ventas': np.random.randint(100, 1000, 20),
    'descuento': np.random.uniform(0, 0.3, 20),
    'fecha': pd.date_range('2023-01-01', periods=20)
})

# Explorando la estructura
print(df_ventas.shape)  # Dimensiones (filas, columnas)
print(df_ventas.info())  # Resumen completo
print(df_ventas.describe())  # Estadísticas descriptivas de columnas numéricas
print_separator()

# Verificando el uso de memoria
print(f"Uso de memoria: {df_ventas.memory_usage(deep=True).sum() / 1024:.2f} KB")

# Optimizando tipos de datos para reducir memoria
df_optimizado = df_ventas.copy()
df_optimizado['región'] = df_optimizado['región'].astype('category')
df_optimizado['producto'] = df_optimizado['producto'].astype('category')

print(f"Memoria después de optimizar: {df_optimizado.memory_usage(deep=True).sum() / 1024:.2f} KB")
print_separator()

# Ejemplo de alineación de datos
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]}, index=['b', 'c', 'd'])

# Suma de DataFrames con índices diferentes
resultado = df1 + df2
print(resultado)
print_separator()

# Diccionario donde cada clave es una columna
datos_columnas = {
    'nombre': ['Juan', 'María', 'Pedro', 'Ana'],
    'puntuación': [85, 92, 78, 96],
    'aprobado': [True, True, True, True]
}

df1 = pd.DataFrame(datos_columnas)
print(df1)
print_separator()

# Lista donde cada diccionario es una fila
datos_filas = [
    {'nombre': 'Juan', 'puntuación': 85, 'aprobado': True},
    {'nombre': 'María', 'puntuación': 92, 'aprobado': True},
    {'nombre': 'Pedro', 'puntuación': 78, 'aprobado': True},
    {'nombre': 'Ana', 'puntuación': 96, 'aprobado': True}
]

df2 = pd.DataFrame(datos_filas)
print(df2)
print_separator()

# Especificando índices personalizados
df3 = pd.DataFrame(datos_columnas, index=['est1', 'est2', 'est3', 'est4'])
print(df3)
print_separator()

# Seleccionando y reordenando columnas
df4 = pd.DataFrame(datos_columnas, columns=['nombre', 'aprobado', 'puntuación'])
print(df4)
print_separator()

# Desde lista de listas
datos_lista = [
    ['Juan', 85, True],
    ['María', 92, True],
    ['Pedro', 78, True],
    ['Ana', 96, True]
]

df5 = pd.DataFrame(datos_lista, columns=['nombre', 'puntuación', 'aprobado'])
print(df5)

# Desde array de NumPy
datos_array = np.array([
    ['Juan', 85, True],
    ['María', 92, True],
    ['Pedro', 78, True],
    ['Ana', 96, True]
])

df6 = pd.DataFrame(datos_array, columns=['nombre', 'puntuación', 'aprobado'])
print(df6)
print_separator()

# Creando desde array homogéneo
array_homogeneo = np.array([[1, 2, 3], [4, 5, 6]])
df_homogeneo = pd.DataFrame(array_homogeneo, columns=['A', 'B', 'C'])
print(df_homogeneo.dtypes)  # Todos serán int64

# Creando desde array mixto
array_mixto = np.array([[1, 'dos', 3.0], [4, 'cinco', 6.0]])
df_mixto = pd.DataFrame(array_mixto, columns=['A', 'B', 'C'])
print(df_mixto.dtypes)  # Todos se convierten a object
print_separator()

# Creando Series individuales
serie_nombres = pd.Series(['Juan', 'María', 'Pedro', 'Ana'], name='nombre')
serie_puntos = pd.Series([85, 92, 78, 96], name='puntuación')
serie_aprobados = pd.Series([True, True, True, True], name='aprobado')

# Combinando Series en un DataFrame
df7 = pd.DataFrame({
    'nombre': serie_nombres,
    'puntuación': serie_puntos,
    'aprobado': serie_aprobados
})
print(df7)

# Alternativa: usando concat
df8 = pd.concat([serie_nombres, serie_puntos, serie_aprobados], axis=1)
print(df8)
print_separator()

# Importando desde CSV
df_csv = pd.read_csv('datos.csv')

# Con opciones adicionales
# df_csv_opciones = pd.read_csv('datos.csv', 
#                              sep=';',              # Separador personalizado
#                              decimal=',',          # Símbolo decimal
#                              encoding='latin1',    # Codificación
#                              header=0,             # Fila de encabezado
#                              index_col='id',       # Columna para usar como índice
#                              skiprows=2,           # Saltar filas iniciales
#                              nrows=100)            # Limitar número de filas

print_separator()

# Importando desde Excel
# df_excel = pd.read_excel('datos.xlsx')

# Con opciones adicionales
# df_excel_opciones = pd.read_excel('datos.xlsx',
#                                  sheet_name='Ventas',    # Nombre de la hoja
#                                  header=1,               # Fila de encabezado
#                                  usecols='A:C',          # Columnas a importar
#                                  skiprows=[0, 2])        # Filas a omitir

print_separator()

# Importando desde JSON
# df_json = pd.read_json('datos.json')

# Con orientación específica
# df_json_orient = pd.read_json('datos.json', orient='records')
print_separator()

# DataFrame vacío
df_vacio = pd.DataFrame()

# DataFrame con estructura pero sin datos
df_estructura = pd.DataFrame(columns=['nombre', 'edad', 'ciudad'])

# Añadiendo filas posteriormente
df_estructura.loc[0] = ['Juan', 30, 'Madrid']
df_estructura.loc[1] = ['María', 25, 'Barcelona']
print(df_estructura)
print_separator()

# Especificando dtypes al crear el DataFrame
df_typed = pd.DataFrame({
    'entero': pd.Series([1, 2, 3], dtype='Int64'),
    'texto': pd.Series(['a', 'b', 'c'], dtype='string'),
    'flotante': pd.Series([1.1, 2.2, 3.3], dtype='Float64'),
    'fecha': pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[ns]'),
    'categoria': pd.Series(['alto', 'medio', 'bajo'], dtype='category')
})

print(df_typed.dtypes)
print_separator()

# Especificando dtypes al crear el DataFrame
df_typed = pd.DataFrame({
    'entero': pd.Series([1, 2, 3], dtype='Int64'),
    'texto': pd.Series(['a', 'b', 'c'], dtype='string'),
    'flotante': pd.Series([1.1, 2.2, 3.3], dtype='Float64'),
    'fecha': pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[ns]'),
    'categoria': pd.Series(['alto', 'medio', 'bajo'], dtype='category')
})

print(df_typed.dtypes)
print_separator()

# DataFrame a diccionario
dict_from_df = df1.to_dict(orient='records')  # Lista de diccionarios (filas)
dict_columns = df1.to_dict()  # Diccionario de series (columnas)

# DataFrame a array NumPy
array_from_df = df1.to_numpy()

# DataFrame a lista de listas
list_from_df = df1.values.tolist()
print_separator()

# Datos aleatorios con NumPy
df_random = pd.DataFrame(
    np.random.randn(5, 3),  # 5 filas, 3 columnas de valores aleatorios
    columns=['A', 'B', 'C'],
    index=pd.date_range('2023-01-01', periods=5)  # Índice de fechas
)
print(df_random)

# Usando la función date_range para crear series temporales
fechas = pd.date_range('2023-01-01', periods=100, freq='D')
valores = np.random.randn(100).cumsum()  # Camino aleatorio
df_timeseries = pd.DataFrame({'fecha': fechas, 'valor': valores})
print(df_timeseries.head())
print_separator()

# Creamos un DataFrame de ejemplo
datos = {
    'producto': ['Laptop', 'Tablet', 'Móvil', 'Monitor', 'Teclado'],
    'precio': [1200, 600, 800, 350, 100],
    'stock': [15, 28, 32, 10, 45],
    'disponible': [True, True, False, True, True]
}

df = pd.DataFrame(datos)
print(df)
print_separator()

# Selección de una columna (devuelve una Series)
precios = df['precio']
print(precios)

# Selección de múltiples columnas (devuelve un DataFrame)
productos_precios = df[['producto', 'precio']]
print(productos_precios)
print_separator()

# Selección de filas por posición usando slicing
primeras_tres_filas = df[0:3]
print(primeras_tres_filas)
print_separator()

# Establecemos un índice personalizado para el ejemplo
df.index = ['A', 'B', 'C', 'D', 'E']
print(df)

# Seleccionar una fila por su etiqueta
fila_b = df.loc['B']
print(fila_b)

# Seleccionar múltiples filas
filas_bd = df.loc[['B', 'D']]
print(filas_bd)

# Seleccionar filas y columnas específicas
subset = df.loc[['A', 'C'], ['producto', 'precio']]
print(subset)

# Seleccionar un rango de filas
rango_filas = df.loc['B':'D']
print(rango_filas)
print_separator()

# Seleccionar filas donde el precio es mayor a 500
productos_caros = df.loc[df['precio'] > 500]
print(productos_caros)

# Seleccionar filas con múltiples condiciones
filtro_complejo = df.loc[(df['precio'] > 300) & (df['stock'] > 20)]
print(filtro_complejo)

# Seleccionar filas que cumplen condiciones y solo algunas columnas
resultado = df.loc[df['disponible'] == True, ['producto', 'precio']]
print(resultado)
print_separator()

# Modificar un valor específico
df.loc['C', 'disponible'] = True

# Modificar múltiples valores
df.loc['A':'B', 'precio'] = [1300, 650]

# Modificar valores basados en una condición
df.loc[df['stock'] < 20, 'stock'] += 5

print(df)
print_separator()

# Seleccionar la primera fila
primera_fila = df.iloc[0]
print(primera_fila)

# Seleccionar múltiples filas por posición
filas_seleccionadas = df.iloc[[0, 2, 4]]
print(filas_seleccionadas)

# Seleccionar filas y columnas por posición
subset_posicional = df.iloc[1:3, 0:2]
print(subset_posicional)

# Seleccionar filas alternas
filas_alternas = df.iloc[::2]  # Filas 0, 2, 4
print(filas_alternas)

# Seleccionar la última fila
ultima_fila = df.iloc[-1]
print(ultima_fila)
print_separator()

# Iterar sobre las primeras 3 filas
for i in range(3):
    print(f"Fila {i}:", df.iloc[i]['producto'], "- Precio:", df.iloc[i]['precio'])
    
# Seleccionar la última columna, independientemente de su nombre
ultima_columna = df.iloc[:, -1]
print(ultima_columna)
print_separator()

# Acceder a un valor específico con .at (usando etiquetas)
precio_tablet = df.at['B', 'precio']
print(f"Precio de la tablet: {precio_tablet}")

# Acceder a un valor específico con .iat (usando posiciones)
stock_movil = df.iat[2, 2]  # Fila 2, columna 2
print(f"Stock de móviles: {stock_movil}")

# Modificar un valor específico
df.at['D', 'stock'] = 12
df.iat[0, 1] = 1250  # Actualizar el precio de la laptop

print(df)
print_separator()

# Ejemplo de rendimiento (no ejecutar en entornos limitados)
import time

# Crear un DataFrame grande
df_grande = pd.DataFrame(np.random.randn(100000, 4), columns=list('ABCD'))

# Medir tiempo con .loc
inicio = time.time()
valor_loc = df_grande.loc[50000, 'B']
tiempo_loc = time.time() - inicio

# Medir tiempo con .at
inicio = time.time()
valor_at = df_grande.at[50000, 'B']
tiempo_at = time.time() - inicio

print(f"Tiempo con .loc: {tiempo_loc:.8f} segundos")
print(f"Tiempo con .at: {tiempo_at:.8f} segundos")
print(f"Mejora de velocidad: {tiempo_loc/tiempo_at:.2f}x")

print_separator()

# Usando query() para filtrar datos
productos_disponibles = df.query('disponible == True and precio < 1000')
print(productos_disponibles)

# Consultas más complejas
resultado_query = df.query('precio > 500 and stock > 15 or producto == "Teclado"')
print(resultado_query)
print_separator()

# Crear una nueva columna con eval()
df.eval('valor_total = precio * stock', inplace=True)
print(df)

# Operaciones más complejas
df.eval('descuento = precio * 0.1 if precio > 500 else precio * 0.05', inplace=True)
print(df[['producto', 'precio', 'descuento']])
print_separator()

# Crear un DataFrame con índice multinivel
idx = pd.MultiIndex.from_tuples([
    ('Electrónica', 'Computadoras', 'Laptop'),
    ('Electrónica', 'Computadoras', 'Tablet'),
    ('Electrónica', 'Teléfonos', 'Móvil'),
    ('Electrónica', 'Periféricos', 'Monitor'),
    ('Electrónica', 'Periféricos', 'Teclado')
], names=['departamento', 'categoría', 'producto'])

df_multi = pd.DataFrame({
    'precio': [1200, 600, 800, 350, 100],
    'stock': [15, 28, 32, 10, 45]
}, index=idx)

print(df_multi)

# Seleccionar todos los productos de una categoría
computadoras = df_multi.xs('Computadoras', level='categoría')
print(computadoras)

# Seleccionar todos los periféricos
perifericos = df_multi.xs(('Electrónica', 'Periféricos'), level=['departamento', 'categoría'])
print(perifericos)
print_separator()

# Ejemplo de vista vs copia
# Vista: modificaciones afectan al original
subset_vista = df.loc['A':'C']
subset_vista.loc['A', 'precio'] = 1400
print(df.loc['A', 'precio'])  # Mostrará 1400

# Copia explícita
subset_copia = df.loc['A':'C'].copy()
subset_copia.loc['A', 'precio'] = 1500
print(df.loc['A', 'precio'])  # Seguirá mostrando 1400

print_separator()

# Acceso a columnas con notación de punto
precios = df.precio
disponibles = df.disponible

# No funciona con nombres de columna que contienen espacios o caracteres especiales
# df.nombre producto  # Esto daría error

print_separator()

# Intentar acceder a una columna que puede no existir
descuento = df.get('descuento_adicional', pd.Series([0, 0, 0, 0, 0]))
print(descuento)
print_separator()

# Creamos un DataFrame de ejemplo
df = pd.DataFrame({
    'producto': ['Laptop', 'Tablet', 'Móvil', 'Monitor', 'Teclado'],
    'precio': [1200, 600, 800, 350, 100],
    'stock': [15, 28, 32, 10, 45]
})

# Añadir una columna con valor constante
df['disponible'] = True
print(df)

# Añadir una columna basada en valores de otras columnas
df['valor_inventario'] = df['precio'] * df['stock']
print(df)
print_separator()

# Añadir una columna con una lista de valores
df['descuento'] = [0.1, 0.05, 0.15, 0.2, 0.1]

# Añadir una columna con valores aleatorios
df['id_producto'] = np.random.randint(1000, 9999, size=len(df))

print(df)
print_separator()

# Añadir múltiples columnas con assign()
df_nuevo = df.assign(
    precio_con_descuento = lambda x: x['precio'] * (1 - x['descuento']),
    ganancia_estimada = lambda x: x['precio_con_descuento'] * 0.3
)

print(df_nuevo.head())
print_separator()

# Encadenamiento de operaciones con assign()
resultado = (df
             .assign(precio_final = lambda x: x['precio'] * (1 - x['descuento']))
             .assign(impuesto = lambda x: x['precio_final'] * 0.21)
             .assign(precio_con_impuesto = lambda x: x['precio_final'] + x['impuesto'])
            )

print(resultado[['producto', 'precio', 'precio_final', 'impuesto', 'precio_con_impuesto']])
print_separator()

# Insertar una columna en la posición 1 (segunda columna)
df.insert(1, 'categoría', ['Informática', 'Informática', 'Telefonía', 'Periféricos', 'Periféricos'])
print(df)
print_separator()

# Eliminar una columna
df_sin_descuento = df.drop('descuento', axis=1)
print(df_sin_descuento.columns)

# Eliminar múltiples columnas
df_simplificado = df.drop(['id_producto', 'valor_inventario', 'descuento'], axis=1)
print(df_simplificado.columns)
print_separator()

# Eliminar columnas modificando el DataFrame original
df.drop('id_producto', axis=1, inplace=True)
print(df.columns)
print_separator()

# Eliminar una columna con del
del df['descuento']
print(df.columns)
print_separator()

# Extraer una columna con pop()
stock_series = df.pop('stock')
print("DataFrame sin la columna 'stock':")
print(df.columns)
print("\nColumna extraída como Series:")
print(stock_series)
print_separator()

# Renombrar columnas específicas
df_renombrado = df.rename(columns={
    'producto': 'nombre_producto',
    'precio': 'precio_unitario',
    'valor_inventario': 'valor_total'
})

print(df_renombrado.columns)
print_separator()

# Renombrar usando una función
df_mayusculas = df.rename(columns=lambda x: x.upper())
print(df_mayusculas.columns)
print_separator()

# Renombrar modificando el DataFrame original
df.rename(columns={'producto': 'nombre_producto'}, inplace=True)
print(df.columns)
print_separator()

# Reemplazar todos los nombres de columnas
df.columns = ['item', 'grupo', 'costo', 'en_stock', 'valor_total']
print(df)
print_separator()

# Añadir prefijo a todas las columnas
df_con_prefijo = df.add_prefix('prod_')
print(df_con_prefijo.columns)

# Añadir sufijo a todas las columnas
df_con_sufijo = df.add_suffix('_item')
print(df_con_sufijo.columns)
print_separator()

# Reordenar columnas específicas
df_reordenado = df[['item', 'costo', 'grupo', 'en_stock', 'valor_total']]
print(df_reordenado.columns)

# Reordenar alfabéticamente
df_alfabetico = df[sorted(df.columns)]
print(df_alfabetico.columns)
print_separator()

# Crear un DataFrame con columnas multinivel
columnas_multi = pd.MultiIndex.from_tuples([
    ('Producto', 'Nombre'),
    ('Producto', 'Categoría'),
    ('Precio', 'Valor'),
    ('Inventario', 'Cantidad'),
    ('Inventario', 'Valor Total')
])

# Asignar las columnas multinivel al DataFrame
df_multi = pd.DataFrame(df.values, index=df.index, columns=columnas_multi)
print(df_multi)
print_separator()

# Aplanar columnas multinivel
df_multi.columns = [f"{col[0]}_{col[1]}" for col in df_multi.columns]
print(df_multi.columns)
print_separator()

# Crear datos para ejemplo de pivot
datos_ventas = pd.DataFrame({
    'fecha': pd.date_range('2023-01-01', periods=10),
    'producto': ['Laptop', 'Tablet', 'Móvil', 'Monitor', 'Teclado'] * 2,
    'ventas': np.random.randint(5, 50, 10)
})

# Transformar usando pivot
tabla_pivot = datos_ventas.pivot(index='fecha', columns='producto', values='ventas')
print(tabla_pivot.head())
print_separator()

# Estandarizar nombres de columnas
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
print_separator()

# Encadenamiento de operaciones
resultado_final = (df
                  .drop(['columna_innecesaria'], axis=1)
                  .rename(columns={'col_a': 'columna_a'})
                  .assign(nueva_col = lambda x: x['columna_a'] * 2)
                 )
print_separator()

# Crear una copia explícita antes de modificar
df_copia = df.copy()
df_copia['nueva_columna'] = 100
print_separator()

# Verificar cambios en la estructura
print(f"Columnas originales: {df.columns.tolist()}")
print(f"Columnas nuevas: {df_nuevo.columns.tolist()}")
print(f"Diferencias: {set(df_nuevo.columns) - set(df.columns)}")
print_separator()
