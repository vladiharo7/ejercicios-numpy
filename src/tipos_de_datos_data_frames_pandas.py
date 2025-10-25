import pandas as pd
import numpy as np

def print_separator():
    print("\n" + "-"*50 + "\n")

# Crear un DataFrame de ejemplo
df = pd.DataFrame({
    'enteros': [1, 2, 3, 4, 5],
    'flotantes': [1.1, 2.2, 3.3, 4.4, 5.5],
    'texto': ['a', 'b', 'c', 'd', 'e'],
    'booleanos': [True, False, True, False, True]
})

# Verificar los tipos de datos
print(df.dtypes)

print_separator()

# Verificar el tipo de una columna específica
print(df['enteros'].dtype)  # int64

print_separator()

df.info()
print_separator()

# Convertir enteros a flotantes
df['enteros_float'] = df['enteros'].astype('float64')

# Convertir flotantes a enteros (se pierden los decimales)
df['flotantes_int'] = df['flotantes'].astype('int64')

# Convertir números a strings
df['enteros_str'] = df['enteros'].astype(str)

print(df.dtypes)
print_separator()

# Convertir múltiples columnas
df = df.astype({
    'enteros': 'float32',
    'texto': 'string'  # Pandas 1.0+ tiene un tipo string dedicado
})

print(df.dtypes)
print_separator()

# DataFrame con valores problemáticos
df_problematico = pd.DataFrame({
    'mixto': ['1', '2', 'tres', '4', '5'],
    'con_nulos': [1.1, 2.2, None, 4.4, 5.5]
})

# Conversión con manejo de errores
try:
    df_problematico['mixto'] = df_problematico['mixto'].astype('int64')
except ValueError as e:
    print(f"Error: {e}")
    # Alternativa: convertir solo lo que se puede
    df_problematico['mixto_seguro'] = pd.to_numeric(df_problematico['mixto'], errors='coerce')
    
print(df_problematico)
print_separator()

# Diferentes opciones de manejo de errores
print("Ignorar errores (mantener originales):")
print(pd.to_numeric(df_problematico['mixto'], errors='ignore'))

print("\nConvertir a NaN los valores problemáticos:")
print(pd.to_numeric(df_problematico['mixto'], errors='coerce'))

print("\nForzar downcast a menor tipo posible:")
print(pd.to_numeric(df_problematico['mixto'], errors='coerce', downcast='integer'))
print_separator()

# Verificar si una columna es numérica
es_numerica = pd.api.types.is_numeric_dtype(df['flotantes'])
print(f"¿La columna 'flotantes' es numérica? {es_numerica}")

# Verificar si una columna es de tipo entero
es_entero = pd.api.types.is_integer_dtype(df['enteros'])
print(f"¿La columna 'enteros' es de tipo entero? {es_entero}")

# Verificar si una columna es de tipo objeto (generalmente strings)
es_objeto = pd.api.types.is_object_dtype(df['texto'])
print(f"¿La columna 'texto' es de tipo objeto? {es_objeto}")
print_separator()

# DataFrame con fechas como strings
df_fechas = pd.DataFrame({
    'fecha_str': ['2023-01-01', '2023-02-01', '2023-03-01']
})

# Convertir strings a fechas
df_fechas['fecha'] = pd.to_datetime(df_fechas['fecha_str'])
print(df_fechas.dtypes)

# Convertir números a categóricos (útil para variables ordinales)
df_categorias = pd.DataFrame({
    'calificacion': [1, 2, 3, 1, 2, 3, 1, 2, 3]
})
df_categorias['calificacion_cat'] = df_categorias['calificacion'].astype('category')
print(df_categorias['calificacion_cat'].cat.categories)
print_separator()

# Crear un archivo CSV de ejemplo
df_ejemplo = pd.DataFrame({
    'enteros': [1, 2, 3],
    'flotantes': [1.1, 2.2, 3.3],
    'texto': ['a', 'b', 'c']
})
df_ejemplo.to_csv('ejemplo.csv', index=False)

# Leer con detección automática de tipos
df_leido = pd.read_csv('ejemplo.csv', dtype_backend='numpy_nullable')
print(df_leido.dtypes)

# También podemos especificar tipos al leer
df_leido_tipos = pd.read_csv('ejemplo.csv', dtype={
    'enteros': 'Int64',  # Tipo entero que admite nulos
    'flotantes': 'float32'
})
print(df_leido_tipos.dtypes)
print_separator()

# Crear un DataFrame con valores faltantes
df = pd.DataFrame({
    'A': [1, np.nan, 3, None, 5],
    'B': ['a', 'b', None, np.nan, 'e'],
    'C': [True, False, None, np.nan, True]
})

print(df)

print(df.dtypes)
print_separator()

# Operaciones estadísticas básicas
print(f"Media de A: {df['A'].mean()}")
print(f"Suma de A: {df['A'].sum()}")
print(f"Mínimo de A: {df['A'].min()}")

print_separator()

# Incluir valores nulos (resultará en NaN)
print(f"Media incluyendo nulos: {df['A'].mean(skipna=False)}")
print_separator()

# Comparaciones con NaN
print(df['A'] > 2)  # Los NaN no son mayores que 2
print(df['A'] == np.nan)  # Importante: esto siempre da False

print_separator()

# Detectar valores faltantes
print("\nDetección de valores faltantes:")
print(df.isna())  # También: df.isnull()

# Detectar valores no faltantes
print("\nDetección de valores no faltantes:")
print(df.notna())  # También: df.notnull()

# Contar valores faltantes por columna
print("\nCantidad de valores faltantes por columna:")
print(df.isna().sum())

# Verificar si alguna columna tiene al menos un valor faltante
print("\n¿Alguna columna tiene valores faltantes?")
print(df.isna().any())

# Verificar si todas las columnas tienen al menos un valor faltante
print("\n¿Todas las columnas tienen valores faltantes?")
print(df.isna().all())
print_separator()

# Filas que tienen al menos un valor faltante
print("\nFilas con al menos un valor faltante:")
print(df[df.isna().any(axis=1)])

# Filas que no tienen ningún valor faltante
print("\nFilas sin valores faltantes:")
print(df[df.notna().all(axis=1)])

# Filas donde la columna A tiene valores faltantes
print("\nFilas donde A tiene valores faltantes:")
print(df[df['A'].isna()])
print_separator()

# Crear un DataFrame para mostrar diferencias
df_diff = pd.DataFrame({
    'con_nan': [1, np.nan, 3],
    'con_none': [1, None, 3]
})

print(df_diff)
print(df_diff.dtypes)
print_separator()

# Comparación de rendimiento (simplificada)
import time

# Crear un DataFrame grande
n = 1000000
df_grande = pd.DataFrame({
    'con_nan': [np.nan] * n,
    'con_none': [None] * n
})

# Medir tiempo para detectar valores faltantes
inicio = time.time()
df_grande['con_nan'].isna().sum()
fin = time.time()
print(f"Tiempo para NaN: {fin - inicio:.6f} segundos")

inicio = time.time()
df_grande['con_none'].isna().sum()
fin = time.time()
print(f"Tiempo para None: {fin - inicio:.6f} segundos")
print_separator()

# Crear DataFrames para comparar uso de memoria
df_int = pd.DataFrame({'enteros': [1, 2, 3, 4, 5]})
df_int_con_na = pd.DataFrame({'enteros_con_na': [1, 2, None, 4, 5]})

print(f"Tipo sin valores nulos: {df_int['enteros'].dtype}")
print(f"Tipo con valores nulos: {df_int_con_na['enteros_con_na'].dtype}")

# Comparar uso de memoria
print(f"Memoria sin nulos: {df_int.memory_usage(deep=True).sum()} bytes")
print(f"Memoria con nulos: {df_int_con_na.memory_usage(deep=True).sum()} bytes")
print_separator()

# Usar tipos que admiten nulos (nullable types)
df_nullable = pd.DataFrame({
    'enteros': pd.Series([1, 2, None, 4, 5], dtype='Int64'),
    'booleanos': pd.Series([True, False, None, True, False], dtype='boolean'),
    'flotantes': pd.Series([1.1, 2.2, None, 4.4, 5.5], dtype='Float64'),
    'strings': pd.Series(['a', 'b', None, 'd', 'e'], dtype='string')
})

print(df_nullable)
print(df_nullable.dtypes)
print_separator()

# Crear un DataFrame para demostrar propagación
df_calc = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8]
})

print("DataFrame original:")
print(df_calc)

print("\nSuma de columnas A + B:")
print(df_calc['A'] + df_calc['B'])

print("\nMultiplicación de columnas A * B:")
print(df_calc['A'] * df_calc['B'])

print("\nOperaciones con métodos agregados:")
print(f"Media de A: {df_calc['A'].mean()}")
print(f"Media de A+B: {(df_calc['A'] + df_calc['B']).mean()}")
print_separator()

# Crear DataFrame para demostrar agrupación
df_grupo = pd.DataFrame({
    'grupo': ['A', 'A', None, 'B', 'B', None],
    'valor': [1, 2, 3, 4, 5, 6]
})

print("Agrupación con valores nulos:")
print(df_grupo.groupby('grupo', dropna=False).sum())
print("\nAgrupación ignorando valores nulos (comportamiento por defecto):")
print(df_grupo.groupby('grupo').sum())
print_separator()

# Crear DataFrames para demostrar joins
df1 = pd.DataFrame({'clave': [1, 2, None, 4], 'valor': ['a', 'b', 'c', 'd']})
df2 = pd.DataFrame({'clave': [1, None, 3, 4], 'otro': ['w', 'x', 'y', 'z']})

print("Inner join con valores nulos:")
print(pd.merge(df1, df2, on='clave', how='inner'))
# Nota: los None no coinciden entre sí

print_separator()

# Crear un DataFrame con datos que se repiten
df = pd.DataFrame({
    'ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Madrid', 'Barcelona', 
               'Valencia', 'Madrid', 'Barcelona', 'Valencia'] * 1000,
    'valoracion': ['Alta', 'Media', 'Baja', 'Media', 'Alta', 
                   'Baja', 'Media', 'Alta', 'Baja'] * 1000
})

# Verificar el uso de memoria inicial
print(f"Memoria antes de conversión: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# Convertir a tipo categórico
df['ciudad_cat'] = df['ciudad'].astype('category')
df['valoracion_cat'] = df['valoracion'].astype('category')

# Verificar el uso de memoria después de la conversión
print(f"Memoria después de conversión: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print_separator()

# Examinar la estructura interna de una columna categórica
print(f"Categorías (valores únicos): {df['ciudad_cat'].cat.categories.tolist()}")
print(f"Códigos (representación interna): {df['ciudad_cat'].cat.codes[:10].tolist()}")
print_separator()

# Crear categorías ordenadas (útil para variables ordinales)
df['valoracion_ordenada'] = pd.Categorical(
    df['valoracion'], 
    categories=['Baja', 'Media', 'Alta'], 
    ordered=True
)

# Comparaciones con categorías ordenadas
print(df[df['valoracion_ordenada'] > 'Baja'].head())
print_separator()

# Filtrado con datos categóricos
# %time filtrado_normal = df[df['ciudad'] == 'Madrid']
# %time filtrado_cat = df[df['ciudad_cat'] == 'Madrid']
# 
# # Agrupación con datos categóricos
# %time agrupado_normal = df.groupby('valoracion').size()
# %time agrupado_cat = df.groupby('valoracion_cat').size()
print_separator()

# Crear una serie de fechas
fechas = pd.Series(pd.date_range('2023-01-01', periods=6, freq='D'))
print(f"Tipo de datos: {fechas.dtype}")
print(fechas)

# Crear un DataFrame con fechas
df_fechas = pd.DataFrame({
    'fecha': pd.date_range('2023-01-01', periods=6, freq='D'),
    'valor': np.random.randn(6)
})
print_separator()

# Extraer componentes de las fechas
df_fechas['año'] = df_fechas['fecha'].dt.year
df_fechas['mes'] = df_fechas['fecha'].dt.month
df_fechas['día'] = df_fechas['fecha'].dt.day
df_fechas['día_semana'] = df_fechas['fecha'].dt.day_name()

print(df_fechas)
print_separator()

# Operaciones aritméticas con fechas
df_fechas['fecha_siguiente'] = df_fechas['fecha'] + pd.Timedelta(days=1)
df_fechas['diferencia_días'] = (df_fechas['fecha_siguiente'] - df_fechas['fecha']).dt.days

# Filtrar por rango de fechas
periodo_medio = df_fechas[(df_fechas['fecha'] >= '2023-01-02') & 
                          (df_fechas['fecha'] <= '2023-01-04')]
print(periodo_medio)
print_separator()

# Crear datos con frecuencia horaria
datos_horarios = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=24, freq='H'),
    'valor': np.random.randn(24)
})

# Resamplear a frecuencia diaria
datos_diarios = datos_horarios.set_index('timestamp').resample('D').mean()
print(datos_diarios)

# Resamplear a frecuencia de 6 horas
datos_6h = datos_horarios.set_index('timestamp').resample('6H').mean()
print(datos_6h)
print_separator()

# Crear un rango de periodos mensuales
periodos = pd.period_range('2023-01', periods=12, freq='M')
print(periodos)

# Crear intervalos de tiempo
intervalos = pd.interval_range(start=pd.Timestamp('2023-01-01'),
                              end=pd.Timestamp('2023-01-05'),
                              freq='D')
print(intervalos)
print_separator()

# Crear una serie con tipo string
textos = pd.Series(['Python', 'Pandas', 'NumPy', 'Matplotlib', None], dtype='string')
print(f"Tipo: {textos.dtype}")

# Ventajas: admite valores nulos sin convertirse a object
print(textos)
print_separator()

# Operaciones con strings
print(textos.str.upper())
print(textos.str.contains('P'))
print(textos.str.len())
print_separator()

# Crear intervalos
intervalos = pd.Series([
    pd.Interval(0, 10),
    pd.Interval(10, 20),
    pd.Interval(20, 30)
])

# Verificar pertenencia
print(5 in intervalos[0])  # True
print(15 in intervalos[0])  # False

# Crear un IntervalIndex para categorización automática
categorias = pd.IntervalIndex.from_breaks([0, 18, 35, 60, 100])
edades = pd.Series([14, 25, 40, 80, 55, 30, 22])
grupos_edad = pd.cut(edades, bins=categorias)
print(grupos_edad)
print_separator()

# Importar tipos de extensión
from pandas import StringDtype, Int64Dtype, BooleanDtype, Float64Dtype

# Crear un DataFrame con tipos de extensión
df_ext = pd.DataFrame({
    'entero': pd.Series([1, 2, None, 4], dtype=Int64Dtype()),
    'flotante': pd.Series([1.1, 2.2, None, 4.4], dtype=Float64Dtype()),
    'texto': pd.Series(['a', 'b', None, 'd'], dtype=StringDtype()),
    'booleano': pd.Series([True, False, None, True], dtype=BooleanDtype())
})

print(df_ext.dtypes)

print_separator()

# Ejemplo simplificado de un tipo personalizado para códigos postales
from pandas.api.extensions import ExtensionArray, ExtensionDtype
import numpy as np

class CodigoPostalArray(ExtensionArray):
    def __init__(self, values):
        self._data = np.array(values, dtype=object)
        
    def __len__(self):
        return len(self._data)
        
    def __getitem__(self, idx):
        return self._data[idx]
        
    @classmethod
    def _from_sequence(cls, scalars):
        return cls(scalars)
        
    def isna(self):
        return np.array([x is None or pd.isna(x) for x in self._data])
        
    # Otros métodos requeridos...

class CodigoPostalDtype(ExtensionDtype):
    name = 'codigo_postal'
    type = str
    na_value = pd.NA
    
    @classmethod
    def construct_array_type(cls):
        return CodigoPostalArray

print_separator()

# Ejemplo con GeoPandas (requiere instalación)
# import geopandas as gpd
# from shapely.geometry import Point

# Simulación simplificada de GeoPandas
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

# Crear un DataFrame con geometrías
df_geo = pd.DataFrame({
    'ciudad': ['Madrid', 'Barcelona', 'Valencia'],
    'geometria': [Point(-3.7, 40.4), Point(2.1, 41.3), Point(-0.3, 39.4)]
})

print(df_geo)
print_separator()

# Crear un DataFrame grande para demostrar optimización
n = 1_000_000
datos_grandes = pd.DataFrame({
    'id': range(n),
    'categoria': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'fecha': np.random.choice(pd.date_range('2020-01-01', '2023-01-01'), n),
    'valor': np.random.randn(n)
})

# Medir memoria inicial
mem_inicial = datos_grandes.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Memoria inicial: {mem_inicial:.2f} MB")

# Optimizar tipos
datos_opt = datos_grandes.copy()
datos_opt['categoria'] = datos_opt['categoria'].astype('category')
datos_opt['fecha'] = pd.to_datetime(datos_opt['fecha'])

# Medir memoria después de optimización
mem_final = datos_opt.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Memoria optimizada: {mem_final:.2f} MB")
print(f"Reducción: {(1 - mem_final/mem_inicial) * 100:.2f}%")
print_separator()

import pandas as pd
import numpy as np

# Crear un DataFrame de ejemplo
df = pd.DataFrame({
    'id': range(1_000_000),
    'entero_grande': np.random.randint(0, 100, 1_000_000),
    'entero_pequeño': np.random.randint(0, 10, 1_000_000),
    'decimal': np.random.random(1_000_000),
    'texto_largo': ['texto_' + str(i % 1000) for i in range(1_000_000)],
    'texto_corto': [['A', 'B', 'C', 'D', 'E'][i % 5] for i in range(1_000_000)]
})

# Analizar el uso de memoria
def analizar_memoria(df):
    uso_memoria = df.memory_usage(deep=True)
    total_memoria = uso_memoria.sum()
    
    print(f"Uso total de memoria: {total_memoria / (1024**2):.2f} MB")
    print("\nUso por columna:")
    for col in uso_memoria.index:
        if col != 'Index':
            porcentaje = uso_memoria[col] / total_memoria * 100
            print(f"{col}: {uso_memoria[col] / (1024**2):.2f} MB ({porcentaje:.1f}%)")
    
    print(f"\nTipos de datos actuales:")
    print(df.dtypes)

analizar_memoria(df)

print_separator()

# Analizar el rango de valores para enteros
def optimizar_enteros(df):
    df_optimizado = df.copy()
    
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Seleccionar el tipo más pequeño que pueda contener el rango
        if col_min >= 0:  # Solo valores positivos
            if col_max < 2**8:
                df_optimizado[col] = df[col].astype('uint8')
            elif col_max < 2**16:
                df_optimizado[col] = df[col].astype('uint16')
            elif col_max < 2**32:
                df_optimizado[col] = df[col].astype('uint32')
        else:  # Valores positivos y negativos
            if col_min >= -2**7 and col_max < 2**7:
                df_optimizado[col] = df[col].astype('int8')
            elif col_min >= -2**15 and col_max < 2**15:
                df_optimizado[col] = df[col].astype('int16')
            elif col_min >= -2**31 and col_max < 2**31:
                df_optimizado[col] = df[col].astype('int32')
    
    return df_optimizado

df_opt = optimizar_enteros(df)
print("Después de optimizar enteros:")
analizar_memoria(df_opt)

print_separator()

# Optimizar flotantes
def optimizar_flotantes(df):
    df_optimizado = df.copy()
    
    for col in df.select_dtypes(include=['float64']).columns:
        # Verificar si float32 ofrece suficiente precisión
        # Esto depende de los requisitos específicos de precisión
        df_optimizado[col] = df[col].astype('float32')
    
    return df_optimizado

df_opt = optimizar_flotantes(df_opt)
print("Después de optimizar flotantes:")
analizar_memoria(df_opt)
print_separator()

# Optimizar texto usando categorías
def optimizar_texto(df, umbral_ratio=0.5):
    df_optimizado = df.copy()
    
    for col in df.select_dtypes(include=['object']).columns:
        # Calcular ratio de valores únicos
        num_valores_unicos = df[col].nunique()
        num_total = len(df[col])
        ratio_unicos = num_valores_unicos / num_total
        
        # Si hay pocos valores únicos en relación al total, usar categoría
        if ratio_unicos < umbral_ratio:
            df_optimizado[col] = df[col].astype('category')
        # Para columnas con muchos valores únicos pero de tipo string
        elif df[col].map(type).eq(str).all():
            df_optimizado[col] = df[col].astype('string')
    
    return df_optimizado

df_opt = optimizar_texto(df_opt)
print("Después de optimizar texto:")
analizar_memoria(df_opt)
print_separator()

def optimizar_dataframe(df, categorias_umbral=0.5, verbose=True):
    """
    Optimiza automáticamente los tipos de datos de un DataFrame para reducir el uso de memoria.
    
    Parámetros:
    - df: DataFrame a optimizar
    - categorias_umbral: Umbral para convertir a categórico (ratio de valores únicos)
    - verbose: Si es True, muestra información sobre los cambios realizados
    
    Retorna:
    - DataFrame optimizado
    """
    inicio_memoria = df.memory_usage(deep=True).sum() / (1024**2)
    if verbose:
        print(f"Memoria inicial: {inicio_memoria:.2f} MB")
    
    df_resultado = df.copy()
    
    # Optimizar enteros
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Seleccionar el tipo más pequeño que pueda contener el rango
        if col_min >= 0:
            if col_max < 2**8:
                df_resultado[col] = df[col].astype('uint8')
            elif col_max < 2**16:
                df_resultado[col] = df[col].astype('uint16')
            elif col_max < 2**32:
                df_resultado[col] = df[col].astype('uint32')
        else:
            if col_min >= -2**7 and col_max < 2**7:
                df_resultado[col] = df[col].astype('int8')
            elif col_min >= -2**15 and col_max < 2**15:
                df_resultado[col] = df[col].astype('int16')
            elif col_min >= -2**31 and col_max < 2**31:
                df_resultado[col] = df[col].astype('int32')
    
    # Optimizar flotantes
    for col in df.select_dtypes(include=['float64']).columns:
        df_resultado[col] = df[col].astype('float32')
    
    # Optimizar texto
    for col in df.select_dtypes(include=['object']).columns:
        # Calcular ratio de valores únicos
        num_valores_unicos = df[col].nunique()
        num_total = len(df[col])
        ratio_unicos = num_valores_unicos / num_total
        
        # Si hay pocos valores únicos en relación al total, usar categoría
        if ratio_unicos < categorias_umbral:
            df_resultado[col] = df[col].astype('category')
        # Para columnas con muchos valores únicos pero de tipo string
        elif df[col].map(type).eq(str).all():
            df_resultado[col] = df[col].astype('string')
    
    # Calcular ahorro de memoria
    final_memoria = df_resultado.memory_usage(deep=True).sum() / (1024**2)
    if verbose:
        print(f"Memoria final: {final_memoria:.2f} MB")
        print(f"Reducción: {(1 - final_memoria/inicio_memoria) * 100:.1f}%")
        print("\nNuevos tipos de datos:")
        print(df_resultado.dtypes)
    
    return df_resultado

# Aplicar optimización automática
df_optimizado = optimizar_dataframe(df)
print_separator()

# Crear un DataFrame con valores nulos
df_con_nulos = pd.DataFrame({
    'entero': [1, 2, None, 4, 5] * 200_000,
    'flotante': [1.1, 2.2, None, 4.4, 5.5] * 200_000,
    'texto': ['a', 'b', None, 'd', 'e'] * 200_000
})

print("Memoria antes de optimizar:")
analizar_memoria(df_con_nulos)

# Optimizar usando tipos que admiten nulos
df_opt_nulos = df_con_nulos.copy()
df_opt_nulos['entero'] = df_con_nulos['entero'].astype('Int32')
df_opt_nulos['flotante'] = df_con_nulos['flotante'].astype('Float32')
df_opt_nulos['texto'] = df_con_nulos['texto'].astype('string')

print("\nMemoria después de optimizar:")
analizar_memoria(df_opt_nulos)
print_separator()

# Ejemplo de carga optimizada desde CSV
dtypes = {
    'id': 'uint32',
    'entero_grande': 'int16',
    'entero_pequeño': 'uint8',
    'decimal': 'float32',
    'texto_corto': 'category'
}

# Simulamos la creación de un CSV
df.to_csv('datos_grandes.csv', index=False)

# Carga optimizada
df_carga_opt = pd.read_csv('datos_grandes.csv', dtype=dtypes)
print("Memoria con carga optimizada:")
analizar_memoria(df_carga_opt)

print_separator()

# Usando dtype_backend en Pandas 2.0
df_auto_opt = pd.read_csv('datos_grandes.csv', dtype_backend='numpy_nullable')
print("Memoria con dtype_backend:")
analizar_memoria(df_auto_opt)

print_separator()

# Identificar columnas que más memoria consumen
def identificar_columnas_pesadas(df, umbral_porcentaje=10):
    uso_memoria = df.memory_usage(deep=True)
    total_memoria = uso_memoria.sum()
    
    columnas_pesadas = []
    for col in uso_memoria.index:
        if col != 'Index':
            porcentaje = uso_memoria[col] / total_memoria * 100
            if porcentaje > umbral_porcentaje:
                columnas_pesadas.append((col, porcentaje, df[col].dtype))
    
    return sorted(columnas_pesadas, key=lambda x: x[1], reverse=True)

# Identificar columnas que más consumen
columnas_pesadas = identificar_columnas_pesadas(df)
print("Columnas que más memoria consumen:")
for col, porcentaje, tipo in columnas_pesadas:
    print(f"{col}: {porcentaje:.1f}% ({tipo})")

# Optimizar solo las columnas más pesadas
df_opt_selectivo = df.copy()
for col, _, _ in columnas_pesadas:
    if col == 'texto_largo':
        df_opt_selectivo[col] = df[col].astype('category')
    elif col == 'decimal':
        df_opt_selectivo[col] = df[col].astype('float32')

print("\nMemoria después de optimización selectiva:")
analizar_memoria(df_opt_selectivo)

print_separator()

# Ejemplo de pérdida de precisión
valor_grande = 16777217  # 2^24 + 1
serie_float64 = pd.Series([valor_grande], dtype='float64')
serie_float32 = pd.Series([valor_grande], dtype='float32')

print(f"float64: {serie_float64[0]}")
print(f"float32: {serie_float32[0]}")
print(f"¿Son iguales?: {serie_float64[0] == serie_float32[0]}")

print_separator()

import gc
import psutil

def mostrar_uso_memoria():
    # Forzar recolección de basura
    gc.collect()
    # Obtener uso de memoria del proceso actual
    proceso = psutil.Process()
    memoria_info = proceso.memory_info()
    return memoria_info.rss / (1024 ** 2)  # En MB

# Ejemplo de uso durante procesamiento
print(f"Memoria antes: {mostrar_uso_memoria():.2f} MB")

# Realizar alguna operación intensiva
df_grande = pd.DataFrame(np.random.randn(1_000_000, 10))
df_grande_procesado = df_grande * 2 + 1

print(f"Memoria después: {mostrar_uso_memoria():.2f} MB")

# Liberar memoria
del df_grande
del df_grande_procesado
gc.collect()

print(f"Memoria tras liberar: {mostrar_uso_memoria():.2f} MB")
print_separator()

# Procesamiento por bloques (chunking)
def procesar_por_bloques(archivo, tamaño_bloque=100_000):
    resultado_total = 0
    
    # Iterar por el archivo en bloques
    for bloque in pd.read_csv(archivo, chunksize=tamaño_bloque):
        # Optimizar tipos para este bloque
        bloque_opt = optimizar_dataframe(bloque, verbose=False)
        
        # Realizar algún procesamiento
        resultado_parcial = bloque_opt['entero_grande'].sum()
        resultado_total += resultado_parcial
        
        # Liberar memoria explícitamente
        del bloque
        del bloque_opt
        gc.collect()
    
    return resultado_total

# Simulación de procesamiento por bloques
# resultado = procesar_por_bloques('datos_grandes.csv')
# print(f"Resultado del procesamiento: {resultado}")

print_separator()