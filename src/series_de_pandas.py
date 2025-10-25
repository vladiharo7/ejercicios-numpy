
import pandas as pd
import numpy as np

def print_separator():
    print("-" * 40)

# Serie con índice implícito (0, 1, 2...)
serie_simple = pd.Series([10, 20, 30])
print(serie_simple)

print_separator()

# Serie con índice explícito
serie_etiquetada = pd.Series([100, 200, 300], index=['a', 'b', 'c'])
print(serie_etiquetada)
print_separator()

# Acceder al array de valores
print("Valores:", serie_etiquetada.values)

# Acceder al índice
print("Índice:", serie_etiquetada.index)
print_separator()

# Serie con índice de fechas
fechas = pd.date_range('20230101', periods=3)
serie_temporal = pd.Series([10, 20, 30], index=fechas)
print(serie_temporal)
print_separator()

# Series con diferentes tipos de datos
serie_int = pd.Series([1, 2, 3])
serie_float = pd.Series([1.1, 2.2, 3.3])
serie_str = pd.Series(['a', 'b', 'c'])
serie_bool = pd.Series([True, False, True])
serie_mixta = pd.Series([1, 'a', 3.14])

print(f"Enteros: {serie_int.dtype}")
print(f"Flotantes: {serie_float.dtype}")
print(f"Strings: {serie_str.dtype}")
print(f"Booleanos: {serie_bool.dtype}")
print(f"Mixta: {serie_mixta.dtype}")
print_separator()

# Usando tipos optimizados
serie_string = pd.Series(['a', 'b', 'c']).astype('string')
serie_int_na = pd.Series([1, None, 3], dtype='Int64')  # Entero que acepta valores nulos
serie_cat = pd.Series(['rojo', 'verde', 'rojo', 'azul']).astype('category')

print(f"String optimizado: {serie_string.dtype}")
print(f"Entero con NA: {serie_int_na.dtype}")
print(f"Categórico: {serie_cat.dtype}")
print_separator()

# Conversión de tipos
serie_numeros = pd.Series(['1', '2', '3'])
print(f"Original: {serie_numeros.dtype}")

# Convertir a entero
serie_enteros = serie_numeros.astype('int64')
print(f"Convertida a entero: {serie_enteros.dtype}")

# Convertir a categórico
serie_cat = serie_numeros.astype('category')
print(f"Convertida a categórico: {serie_cat.dtype}")
print_separator()

# Serie con valores nulos
serie_con_nulos = pd.Series([1, None, 3, None, 5])

# Verificar valores nulos
print("¿Hay valores nulos?")
print(serie_con_nulos.isna())

# Contar valores nulos
print(f"Número de valores nulos: {serie_con_nulos.isna().sum()}")

# Verificar si los valores son numéricos
print("¿Son numéricos?")
print(pd.Series([1, 2, '3']).apply(lambda x: pd.api.types.is_numeric_dtype(type(x))))
print_separator()

# Crear una Serie desde un array de NumPy
array_numpy = np.array([10, 20, 30])
serie_numpy = pd.Series(array_numpy)

# Verificar el tipo del objeto interno
print(f"Tipo del array interno: {type(serie_numpy.array)}")

# Convertir Serie a array de NumPy
array_convertido = serie_numpy.to_numpy()
print(f"Array convertido: {array_convertido}")
print_separator()

# Creación básica desde una lista
numeros = [10, 20, 30, 40, 50]
serie_numeros = pd.Series(numeros)
print(serie_numeros)
print_separator()

# Creación desde un diccionario
datos_ciudades = {'Madrid': 3.2, 'Barcelona': 1.6, 'Valencia': 0.8, 'Sevilla': 0.7}
poblacion = pd.Series(datos_ciudades)
print(poblacion)
print_separator()

# Selección y reordenación de elementos del diccionario
ciudades_seleccionadas = ['Barcelona', 'Madrid', 'Sevilla']
poblacion_seleccionada = pd.Series(datos_ciudades, index=ciudades_seleccionadas)
print(poblacion_seleccionada)
print_separator()

# Creación desde un escalar
indice = ['a', 'b', 'c', 'd', 'e']
serie_constante = pd.Series(100, index=indice)
print(serie_constante)
print_separator()

# Desde array de NumPy básico
array_np = np.array([1, 2, 3, 4, 5])
serie_np = pd.Series(array_np)
print(serie_np)

# Desde array generado con funciones de NumPy
serie_lineal = pd.Series(np.linspace(0, 10, 5))  # 5 valores equidistantes entre 0 y 10
print("\nSerie con valores lineales:")
print(serie_lineal)

# Desde array con distribución aleatoria
serie_aleatoria = pd.Series(np.random.randn(5))  # 5 valores de distribución normal
print("\nSerie con valores aleatorios:")
print(serie_aleatoria)
print_separator()


# Desde array de NumPy básico
array_np = np.array([1, 2, 3, 4, 5])
serie_np = pd.Series(array_np)
print(serie_np)

# Desde array generado con funciones de NumPy
serie_lineal = pd.Series(np.linspace(0, 10, 5))  # 5 valores equidistantes entre 0 y 10
print("\nSerie con valores lineales:")
print(serie_lineal)

# Desde array con distribución aleatoria
serie_aleatoria = pd.Series(np.random.randn(5))  # 5 valores de distribución normal
print("\nSerie con valores aleatorios:")
print(serie_aleatoria)
print_separator()

# Desde un archivo CSV (primera columna como datos)
# Suponiendo un archivo 'datos.csv' con una columna de valores
# serie_csv = pd.read_csv('datos.csv', squeeze=True)

# Desde un archivo Excel (primera columna como datos)
# serie_excel = pd.read_excel('datos.xlsx', sheet_name='Hoja1', squeeze=True)

print_separator()

# Método moderno en Pandas 2
# serie_csv = pd.read_csv('datos.csv').iloc[:, 0]  # Primera columna como Serie
print_separator()

# Desde una comprensión de lista
serie_cuadrados = pd.Series([x**2 for x in range(1, 6)])
print(serie_cuadrados)

# Usando un generador con función map
serie_cubos = pd.Series(map(lambda x: x**3, range(1, 6)))
print("\nSerie de cubos:")
print(serie_cubos)
print_separator()

# Serie original
serie_base = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])

# Creación por selección
serie_seleccionada = serie_base[['a', 'c', 'e']]
print(serie_seleccionada)

# Creación por transformación
serie_transformada = serie_base.apply(lambda x: x * 2)
print("\nSerie transformada:")
print(serie_transformada)
print_separator()

# Serie con fechas como índice
fechas = pd.date_range(start='2023-01-01', periods=5, freq='D')
serie_temporal = pd.Series(range(100, 600, 100), index=fechas)
print(serie_temporal)

# Serie con períodos de tiempo
periodos = pd.period_range(start='2023Q1', periods=4, freq='Q')  # Trimestres
serie_trimestral = pd.Series([1000, 1200, 1100, 1300], index=periodos)
print("\nSerie trimestral:")
print(serie_trimestral)
print_separator()

# Datos con categorías repetidas
colores = ['rojo', 'verde', 'azul', 'rojo', 'verde', 'rojo', 'azul']

# Creación como categorías
serie_categorica = pd.Series(colores).astype('category')
print(serie_categorica)
print("\nCategorías:")
print(serie_categorica.cat.categories)
print("\nCódigos:")
print(serie_categorica.cat.codes)  # Representación numérica interna
print_separator()

# Función para generar secuencia de Fibonacci
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Crear Serie con los primeros 8 números de Fibonacci
serie_fibonacci = pd.Series(list(fibonacci(8)))
print(serie_fibonacci)

# Función para generar datos con ruido
def datos_con_ruido(base, ruido, n):
    return [base + i + np.random.normal(0, ruido) for i in range(n)]

# Serie con tendencia lineal y ruido aleatorio
serie_tendencia = pd.Series(datos_con_ruido(100, 5, 5))
print("\nSerie con tendencia y ruido:")
print(serie_tendencia)
print_separator()

# Crear un DataFrame simple
df = pd.DataFrame({
    'Nombre': ['Ana', 'Carlos', 'Elena', 'David'],
    'Edad': [28, 35, 42, 31],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla']
})

# Extraer una columna como Serie
edades = df['Edad']
print(edades)
print(f"\nTipo: {type(edades)}")
print_separator()

# Crear dos Series
s1 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

# Operaciones básicas
suma = s1 + s2
resta = s1 - s2
multiplicacion = s1 * s2
division = s1 / s2

print("Suma:")
print(suma)
print("\nMultiplicación:")
print(multiplicacion)

print_separator()

# Series con índices diferentes
s3 = pd.Series([100, 200, 300], index=['a', 'b', 'e'])
resultado = s1 + s3

print(resultado)
print_separator()

# Multiplicar toda la Serie por un escalar
s_escalada = s1 * 10
print(s_escalada)

# Elevar al cuadrado cada elemento
s_cuadrado = s1 ** 2
print("\nCuadrados:")
print(s_cuadrado)
print_separator()

# Crear una Serie con datos numéricos
datos = pd.Series([12, 45, 23, 67, 34, 89, 56, 12])

# Estadísticas descriptivas básicas
print(f"Media: {datos.mean()}")
print(f"Mediana: {datos.median()}")
print(f"Desviación estándar: {datos.std()}")
print(f"Valor mínimo: {datos.min()}")
print(f"Valor máximo: {datos.max()}")

# Resumen estadístico completo
print("\nResumen estadístico:")
print(datos.describe())
print_separator()

# Cuantiles personalizados
print(datos.quantile([0.1, 0.5, 0.9]))

# Correlación y covarianza entre Series
s4 = pd.Series([14, 48, 26, 70, 38, 92, 60, 15])
print(f"\nCorrelación: {datos.corr(s4)}")
print(f"Covarianza: {datos.cov(s4)}")
print_separator()

# Crear una Serie de ejemplo
ventas = pd.Series([100, 150, 200, 250, 300], 
                  index=pd.date_range('2023-01-01', periods=5, freq='M'))
print("Ventas mensuales:")
print(ventas)

# Suma acumulativa
print("\nVentas acumuladas:")
print(ventas.cumsum())

# Producto acumulativo
print("\nCrecimiento acumulado (si cada valor fuera una tasa):")
print(ventas.cumprod() / 100)

# Máximo y mínimo acumulados
print("\nMáximo acumulado:")
print(ventas.cummax())
print_separator()

# Aplicar una función a cada elemento
numeros = pd.Series([1, 2, 3, 4, 5])

# Usando apply con una función lambda
raices = numeros.apply(lambda x: np.sqrt(x))
print("Raíces cuadradas:")
print(raices)

# Usando apply con una función definida
def categorizar(valor):
    if valor < 3:
        return "bajo"
    elif valor < 5:
        return "medio"
    else:
        return "alto"

categorias = numeros.apply(categorizar)
print("\nCategorías:")
print(categorias)
print_separator()

# Usando map con un diccionario
mapeo = {1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco"}
numeros_texto = numeros.map(mapeo)
print("Números en texto:")
print(numeros_texto)

# Usando map con una Serie
conversion = pd.Series(["A", "B", "C"], index=[1, 3, 5])
resultado_map = numeros.map(conversion)
print("\nMapeo con otra Serie:")
print(resultado_map)  # Notar que solo mapea los índices que coinciden
print_separator()
# Crear una Serie con datos variados
calificaciones = pd.Series([85, 92, 78, 65, 98, 72, 88, 95], 
                          index=['Ana', 'Carlos', 'Elena', 'David', 
                                'Laura', 'Miguel', 'Sofía', 'Pablo'])

# Filtrado con condiciones booleanas
aprobados = calificaciones[calificaciones >= 70]
print("Estudiantes aprobados:")
print(aprobados)

# Filtrado múltiple
destacados = calificaciones[(calificaciones >= 90) & (calificaciones < 100)]
print("\nEstudiantes destacados:")
print(destacados)

# Usando query (más legible para condiciones complejas)
recuperacion = calificaciones.loc[calificaciones.between(65, 69)]
print("\nEstudiantes en recuperación:")
print(recuperacion)
print_separator()

# Serie con valores faltantes
datos_incompletos = pd.Series([10, np.nan, 30, np.nan, 50, 60])

# Detectar valores nulos
print("¿Dónde hay valores nulos?")
print(datos_incompletos.isna())

# Eliminar valores nulos
print("\nSerie sin valores nulos:")
print(datos_incompletos.dropna())

# Rellenar valores nulos
print("\nRellenando con ceros:")
print(datos_incompletos.fillna(0))

# Rellenar con el valor anterior o siguiente
print("\nRellenando con el valor anterior:")
print(datos_incompletos.fillna(method='ffill'))

print("\nRellenando con el valor siguiente:")
print(datos_incompletos.fillna(method='bfill'))

# Interpolación
print("\nInterpolación lineal:")
print(datos_incompletos.interpolate())
print_separator()

# Crear una Serie temporal
fechas = pd.date_range('2023-01-01', periods=6, freq='M')
ventas_mensuales = pd.Series([12000, 15000, 14000, 16500, 18000, 17500], index=fechas)
print("Ventas mensuales 2023:")
print(ventas_mensuales)

# Resample (cambiar la frecuencia)
ventas_trimestrales = ventas_mensuales.resample('Q').sum()
print("\nVentas trimestrales:")
print(ventas_trimestrales)

# Desplazamiento temporal
print("\nVentas con retraso de 1 mes:")
print(ventas_mensuales.shift(1))  # Desplaza valores 1 período hacia adelante

# Diferencia porcentual
print("\nCambio porcentual mes a mes:")
print(ventas_mensuales.pct_change() * 100)

# Promedio móvil (tendencia)
print("\nPromedio móvil (2 meses):")
print(ventas_mensuales.rolling(window=2).mean())
print_separator()


# Crear una Serie categórica
colores = pd.Series(['rojo', 'verde', 'azul', 'rojo', 'verde', 'amarillo', 'rojo'])
colores_cat = colores.astype('category')

# Obtener frecuencias
print("Frecuencia de cada color:")
print(colores_cat.value_counts())

# Obtener códigos numéricos
print("\nCódigos internos:")
print(colores_cat.cat.codes)

# Añadir nuevas categorías
colores_cat = colores_cat.cat.add_categories(['naranja', 'morado'])
print("\nCategorías disponibles:")
print(colores_cat.cat.categories)

# Eliminar categorías no utilizadas
colores_reducidas = colores_cat.cat.remove_unused_categories()
print("\nCategorías después de eliminar no utilizadas:")
print(colores_reducidas.cat.categories)
print_separator()

# Crear una Serie desordenada
datos = pd.Series([30, 15, 45, 10, 25], index=['c', 'e', 'a', 'd', 'b'])
print("Serie original:")
print(datos)

# Ordenar por valor
ordenada_valor = datos.sort_values()
print("\nOrdenada por valor:")
print(ordenada_valor)

# Ordenar por índice
ordenada_indice = datos.sort_index()
print("\nOrdenada por índice:")
print(ordenada_indice)

# Ordenar descendentemente
ordenada_desc = datos.sort_values(ascending=False)
print("\nOrdenada descendente por valor:")
print(ordenada_desc)
print_separator()

# Crear dos Series
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['d', 'e', 'f'])

# Concatenación
concatenada = pd.concat([s1, s2])
print("Series concatenadas:")
print(concatenada)

# Combinación con operación específica
s3 = pd.Series([10, 20, 30], index=['a', 'b', 'g'])
combinada = s1.combine(s3, lambda x, y: x if pd.isna(y) else y)
print("\nCombinación selectiva:")
print(combinada)

# Actualización (update)
s4 = pd.Series([100, 200], index=['a', 'c'])
s1_copia = s1.copy()
s1_copia.update(s4)
print("\nSerie actualizada:")
print(s1_copia)
print_separator()

# Crear una Serie de ejemplo
datos = pd.Series([10, 20, 30, 40], index=['w', 'x', 'y', 'z'])

# Convertir a lista
lista = datos.tolist()
print(f"Lista: {lista}")

# Convertir a diccionario
diccionario = datos.to_dict()
print(f"Diccionario: {diccionario}")

# Convertir a DataFrame
df = datos.to_frame(name='Valores')
print("\nDataFrame:")
print(df)

# Convertir a JSON
json_str = datos.to_json()
print(f"\nJSON: {json_str}")

# Convertir a array de NumPy
array = datos.to_numpy()
print(f"NumPy array: {array}")
print_separator()

# Crear una Serie con índice personalizado
ventas = pd.Series([1200, 1500, 900, 1800, 1350],
                  index=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes'])
print(ventas)
print_separator()

# Acceder a un único elemento
print("Ventas del martes:", ventas.loc['Martes'])

# Acceder a múltiples elementos
print("\nVentas de inicio de semana:")
print(ventas.loc[['Lunes', 'Martes']])

# Acceder a un rango de etiquetas (inclusivo)
print("\nVentas de mitad de semana:")
print(ventas.loc['Martes':'Jueves'])
print_separator()

# Acceder al primer elemento (posición 0)
print("Primer día:", ventas.iloc[0])

# Acceder a múltiples posiciones
print("\nPrimero y último día:")
print(ventas.iloc[[0, 4]])

# Acceder a un rango de posiciones (el último no inclusivo)
print("\nTres días centrales:")
print(ventas.iloc[1:4])
print_separator()

# Si el índice es de etiquetas, accede por etiqueta
print("Usando corchetes con etiqueta:", ventas['Miércoles'])

# Si usamos un slice, se interpreta como etiquetas
print("\nSlice con corchetes (por etiqueta):")
print(ventas['Lunes':'Miércoles'])

# Acceso a múltiples elementos
print("\nLista de etiquetas con corchetes:")
print(ventas[['Lunes', 'Viernes']])
print_separator()

# Serie con índice numérico
numeros = pd.Series([10, 20, 30, 40, 50], index=[1, 2, 3, 4, 5])
print(numeros)

# Esto accede por posición, no por etiqueta
# print("\nAcceso por posición:", numeros[0])  # Error! No existe índice 0

# Para evitar confusiones, mejor usar .loc y .iloc
print("Primer elemento con .iloc:", numeros.iloc[0])  # Posición 0
print("Elemento con índice 1 con .loc:", numeros.loc[1])  # Etiqueta 1
print_separator()

# Filtrar elementos que cumplen una condición
dias_buenos = ventas[ventas > 1300]
print("Días con ventas superiores a 1300:")
print(dias_buenos)

# Combinar múltiples condiciones
dias_especificos = ventas[(ventas >= 1200) & (ventas <= 1500)]
print("\nDías con ventas entre 1200 y 1500:")
print(dias_especificos)

# Usar funciones para condiciones más complejas
dias_pares = ventas[ventas.index.str.len() > 6]  # Días con nombre largo
print("\nDías con nombre largo:")
print(dias_pares)
print_separator()

# Serie original
temperaturas = pd.Series([22, 25, 30], index=['Lunes', 'Miércoles', 'Viernes'])
print("Temperaturas originales:")
print(temperaturas)

# Reindexar con todos los días de la semana
dias_completos = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas_completas = temperaturas.reindex(dias_completos)
print("\nTemperaturas reindexadas (con NaN):")
print(temperaturas_completas)
print_separator()

# Rellenar con un valor constante
temp_rellenadas = temperaturas.reindex(dias_completos, fill_value=0)
print("Rellenadas con cero:")
print(temp_rellenadas)

# Propagar el último valor válido (forward fill)
temp_ffill = temperaturas.reindex(dias_completos, method='ffill')
print("\nPropagación hacia adelante:")
print(temp_ffill)

# Propagar el siguiente valor válido (backward fill)
temp_bfill = temperaturas.reindex(dias_completos, method='bfill')
print("\nPropagación hacia atrás:")
print(temp_bfill)
print_separator()

# Crear una Serie con índice numérico
alturas = pd.Series([1.70, 1.90], index=[0, 4])
print("Alturas originales:")
print(alturas)

# Reindexar con interpolación lineal
alturas_completas = alturas.reindex(range(6), method='linear')
print("\nAlturas interpoladas:")
print(alturas_completas)
print_separator()

# Reordenar los elementos
dias_invertidos = ['Domingo', 'Sábado', 'Viernes', 'Jueves', 'Miércoles', 'Martes', 'Lunes']
temperaturas_invertidas = temperaturas.reindex(dias_invertidos)
print("Temperaturas en orden inverso:")
print(temperaturas_invertidas)
print_separator()

# Dos Series con índices diferentes
serie1 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
serie2 = pd.Series([100, 200, 300, 400], index=['b', 'c', 'd', 'e'])

print("Serie 1:")
print(serie1)
print("\nSerie 2:")
print(serie2)

# Operación con alineación automática
suma = serie1 + serie2
print("\nSuma con alineación automática:")
print(suma)
print_separator()

# Rellenar valores faltantes con ceros antes de operar
suma_rellenada = serie1.add(serie2, fill_value=0)
print("Suma rellenando con ceros:")
print(suma_rellenada)

# Usar solo las etiquetas que coinciden (intersección)
indices_comunes = serie1.index.intersection(serie2.index)
suma_interseccion = serie1[indices_comunes] + serie2[indices_comunes]
print("\nSuma solo de índices comunes:")
print(suma_interseccion)
print_separator()

# Diferentes operaciones con alineación
resta = serie1.sub(serie2, fill_value=0)
multiplicacion = serie1.mul(serie2, fill_value=1)
division = serie1.div(serie2, fill_value=1)

print("Resta con alineación:")
print(resta)
print("\nMultiplicación con alineación:")
print(multiplicacion)
print_separator()

# Series con datos de ventas por producto
ventas_enero = pd.Series({'Manzanas': 100, 'Peras': 120, 'Naranjas': 80})
ventas_febrero = pd.Series({'Peras': 130, 'Naranjas': 90, 'Plátanos': 110})

# Calcular ventas totales
ventas_totales = ventas_enero.add(ventas_febrero, fill_value=0)
print("Ventas totales por producto:")
print(ventas_totales)

# Calcular promedio (teniendo en cuenta solo los meses con ventas)
ventas_promedio = (ventas_enero + ventas_febrero) / 2
print("\nPromedio incorrecto (con NaN):")
print(ventas_promedio)

# Forma correcta de calcular el promedio
ventas_promedio_correcto = ventas_totales / 2
print("\nPromedio correcto:")
print(ventas_promedio_correcto)
print_separator()

# Crear Series con índices jerárquicos
idx1 = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)])
idx2 = pd.MultiIndex.from_tuples([('A', 1), ('B', 1), ('B', 2)])

s1 = pd.Series([1, 2, 3], index=idx1)
s2 = pd.Series([10, 20, 30], index=idx2)

print("Serie con índice jerárquico 1:")
print(s1)
print("\nSerie con índice jerárquico 2:")
print(s2)

# Operación con alineación automática
resultado = s1 + s2
print("\nSuma con alineación multinivel:")
print(resultado)
print_separator()

# Reindexar una Serie con índice jerárquico
nuevo_idx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2), ('C', 1)])
s1_reindexada = s1.reindex(nuevo_idx)
print("Serie reindexada con multinivel:")
print(s1_reindexada)
print_separator()

# Ejemplo: Análisis de ventas mensuales por producto
ventas_q1 = pd.Series({
    'Producto A': [100, 120, 110],
    'Producto B': [90, 95, 100],
    'Producto C': [80, 85, 90]
}, index=['Enero', 'Febrero', 'Marzo'])

ventas_q2 = pd.Series({
    'Producto A': [115, 125, 130],
    'Producto B': [105, 110, 115],
    'Producto D': [70, 75, 80]
}, index=['Abril', 'Mayo', 'Junio'])

# Combinar datos de ambos trimestres
ventas_semestre = pd.concat([ventas_q1, ventas_q2])

# Reindexar para incluir todos los productos en todos los meses
todos_productos = ['Producto A', 'Producto B', 'Producto C', 'Producto D']
todos_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']

# Crear una matriz completa de ventas (productos x meses)
ventas_completas = pd.DataFrame(index=todos_meses, columns=todos_productos)

# Rellenar con los datos disponibles
for mes in todos_meses:
    for producto in todos_productos:
        if mes in ventas_semestre.index and producto in ventas_semestre[mes]:
            ventas_completas.loc[mes, producto] = ventas_semestre.loc[mes, producto]
        else:
            ventas_completas.loc[mes, producto] = 0

print("Matriz completa de ventas:")
print(ventas_completas)
print_separator()


