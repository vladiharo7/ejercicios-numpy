import pandas as pd
import numpy as np

def separator_line():
    print("\n" + "-"*50 + "\n")

# Creamos un DataFrame de ejemplo con datos de ventas
datos = {
    'producto': ['Laptop', 'Teléfono', 'Tablet', 'Monitor', 'Teclado'],
    'precio': [1200, 800, 350, 250, 100],
    'stock': [15, 25, 40, 30, 50],
    'categoría': ['Computación', 'Móviles', 'Móviles', 'Periféricos', 'Periféricos']
}

df = pd.DataFrame(datos)
print(df)

separator_line()

# Crear una condición booleana
productos_caros = df['precio'] > 500

# Ver la serie booleana resultante
print(productos_caros)

# Aplicar el filtro al DataFrame
print(df[productos_caros])

# Todo en una sola línea
print(df[df['precio'] > 500])
separator_line()

# Productos caros con poco stock
filtro = (df['precio'] > 500) & (df['stock'] < 20)
print(df[filtro])

# Productos de la categoría 'Móviles' o con precio menor a 200
filtro = (df['categoría'] == 'Móviles') | (df['precio'] < 200)
print(df[filtro])

# Productos que NO son periféricos
filtro = ~(df['categoría'] == 'Periféricos')
print(df[filtro])
separator_line()

# Productos cuyo nombre contiene la letra 'o'
filtro = df['producto'].str.contains('o')
print(df[filtro])

separator_line()
# Equivalente a df[df['precio'] > 500]
resultado = df.query('precio > 500')
print(resultado)

# Condiciones múltiples
resultado = df.query('precio > 300 and stock < 30')
print(resultado)
separator_line()

# Usando operadores lógicos
resultado = df.query('categoría == "Móviles" or precio < 200')
print(resultado)

# Usando not
resultado = df.query('not (categoría == "Periféricos")')
print(resultado)
separator_line()

# DataFrame con una columna que tiene espacios
df2 = pd.DataFrame({
    'nombre producto': ['A', 'B', 'C'],
    'precio unitario': [10, 20, 30]
})

# Usando backticks para referenciar columnas con espacios
resultado = df2.query('`precio unitario` > 15')
print(resultado)
separator_line()

# Definimos variables externas
precio_limite = 400
categorias_interes = ['Móviles', 'Computación']

# Usamos las variables en la consulta
resultado = df.query('precio < @precio_limite and categoría in @categorias_interes')
print(resultado)
separator_line()

# Para DataFrames grandes, query() puede ser más eficiente
df_grande = pd.DataFrame({
    'A': np.random.randint(0, 100, size=100000),
    'B': np.random.randint(0, 100, size=100000)
})

# Comparación de rendimiento (no se muestra el resultado por brevedad)
# %timeit 
df_grande[df_grande['A'] > 50]
# %timeit 
df_grande.query('A > 50')
separator_line()

# Eliminar registros con valores extremos
datos_limpios = df.query('precio > 0 and precio < 10000')
separator_line()

# Segmentar por categoría
moviles = df.query('categoría == "Móviles"')
perifericos = df.query('categoría == "Periféricos"')

separator_line()

# Calcular el precio promedio de productos con stock bajo
precio_promedio = df.query('stock < 20')['precio'].mean()

separator_line()

# Exportar solo productos de alta rotación
df.query('stock < 30').to_csv('productos_alta_rotacion.csv')

separator_line()

# Creamos un DataFrame con datos de empleados
datos = {
    'nombre': ['Ana', 'Carlos', 'Elena', 'David', 'Beatriz'],
    'departamento': ['Ventas', 'IT', 'Marketing', 'IT', 'Finanzas'],
    'salario': [45000, 60000, 52000, 58000, 63000],
    'experiencia': [3, 5, 2, 7, 4]
}

df = pd.DataFrame(datos)
print(df)
separator_line()
# Reemplazar salarios menores a 55000 con NaN
resultado = df['salario'].where(df['salario'] >= 55000)
print(resultado)
separator_line()

# Reemplazar salarios menores a 55000 con 50000
resultado = df['salario'].where(df['salario'] >= 55000, 50000)
print(resultado)
separator_line()

# Aplicar una condición a todo el DataFrame
umbral_experiencia = 4
resultado = df.where(df['experiencia'] >= umbral_experiencia)
print(resultado)
separator_line()

# Condiciones específicas por columna
condiciones = {
    'salario': df['salario'] > 55000,
    'experiencia': df['experiencia'] >= 5
}

# Crear un DataFrame con las mismas dimensiones
mascara = pd.DataFrame(condiciones, index=df.index)

# Aplicar where con la máscara
resultado = df.where(mascara, "No cumple")
print(resultado)
separator_line()

# Reemplazar salarios mayores o iguales a 55000 con NaN
resultado = df['salario'].mask(df['salario'] >= 55000)
print(resultado)

# Reemplazar salarios mayores o iguales a 55000 con 55000 (aplicar un tope)
resultado = df['salario'].mask(df['salario'] >= 55000, 55000)
print(resultado)
separator_line()
# Censurar nombres de empleados con salarios altos
resultado = df['nombre'].mask(df['salario'] > 60000, "***")
print(resultado)
separator_line()

# Filtrar empleados de departamentos específicos
departamentos_objetivo = ['IT', 'Finanzas']
filtro = df['departamento'].isin(departamentos_objetivo)
print(filtro)
print(df[filtro])
separator_line()

# Comparación con operador de igualdad (solo funciona para un valor)
print(df[df['departamento'] == 'IT'])

# Con isin() podemos verificar múltiples valores a la vez
print(df[df['departamento'].isin(['IT', 'Finanzas'])])
separator_line()

# Empleados que NO están en IT ni Finanzas
print(df[~df['departamento'].isin(['IT', 'Finanzas'])])
separator_line()

# Crear un DataFrame con valores a buscar
valores_buscar = pd.DataFrame({
    'nombre': ['Ana', 'David'],
    'departamento': ['Ventas', 'Marketing']
})

# Buscar coincidencias exactas en ambas columnas
resultado = df.isin(valores_buscar)
print(resultado)
separator_line()

# Identificar valores atípicos y reemplazarlos
salario_promedio = df['salario'].mean()
desviacion = df['salario'].std()
limite_superior = salario_promedio + 1.5 * desviacion

# Reemplazar salarios atípicos con el límite
df['salario_ajustado'] = df['salario'].mask(df['salario'] > limite_superior, limite_superior)
print(df[['salario', 'salario_ajustado']])
separator_line()

# Categorizar empleados según experiencia y salario
condicion_senior = (df['experiencia'] >= 5) & (df['salario'] >= 55000)
condicion_mid = (df['experiencia'] >= 3) & (df['salario'] >= 45000)

# Crear una nueva columna con categorías
df['nivel'] = 'Junior'  # Valor predeterminado
df['nivel'] = df['nivel'].mask(condicion_mid, 'Mid-level')
df['nivel'] = df['nivel'].mask(condicion_senior, 'Senior')
print(df)
separator_line()

# Reemplazar valores negativos con cero
df_numerico = pd.DataFrame({
    'A': [1, -2, 3, -4],
    'B': [-5, 6, -7, 8]
})
df_limpio = df_numerico.mask(df_numerico < 0, 0)
print(df_limpio)
separator_line()

# Anonimizar información personal
df_personal = pd.DataFrame({
    'nombre': ['Juan Pérez', 'María López', 'Carlos Ruiz'],
    'email': ['juan@ejemplo.com', 'maria@ejemplo.com', 'carlos@ejemplo.com'],
    'edad': [34, 28, 45]
})

# Ocultar emails excepto para personas mayores de 30
df_personal['email'] = df_personal['email'].mask(df_personal['edad'] <= 30, '***@***.com')
print(df_personal)
separator_line()

# Verificar si los departamentos son válidos
departamentos_validos = ['Ventas', 'IT', 'Marketing', 'Finanzas', 'RRHH']
df['departamento_valido'] = df['departamento'].isin(departamentos_validos)
print(df)
separator_line()

# Aplicar bonificación según departamento
bonificaciones = {'Ventas': 0.1, 'IT': 0.08, 'Marketing': 0.07, 'Finanzas': 0.06}
df['bonificacion'] = 0

for depto, porcentaje in bonificaciones.items():
    mascara = df['departamento'] == depto
    df['bonificacion'] = df['bonificacion'].mask(mascara, df['salario'] * porcentaje)

print(df[['nombre', 'departamento', 'salario', 'bonificacion']])
separator_line()

# Creamos un DataFrame con datos de productos
datos = {
    'producto': ['Monitor', 'Teclado', 'Mouse', 'Laptop', 'Tablet'],
    'precio': [250, 80, 45, 1200, 350],
    'stock': [30, 120, 150, 15, 40],
    'fecha_ingreso': pd.to_datetime(['2023-05-15', '2023-02-10', 
                                     '2023-03-22', '2023-01-05', '2023-04-18'])
}

df = pd.DataFrame(datos)
print(df)
separator_line()

# Ordenar por precio (ascendente por defecto)
df_ordenado = df.sort_values('precio')
print(df_ordenado)

# Ordenar por precio en orden descendente
df_ordenado_desc = df.sort_values('precio', ascending=False)
print(df_ordenado_desc)
separator_line()

# Ordenar primero por stock (ascendente) y luego por precio (descendente)
df_multi_orden = df.sort_values(['stock', 'precio'], ascending=[True, False])
print(df_multi_orden)
separator_line()

# Ordenar por stock (ascendente) y precio (descendente)
df_orden_mixto = df.sort_values(['stock', 'precio'], ascending=[True, False])
print(df_orden_mixto)
separator_line()

# Creamos un DataFrame con algunos valores nulos
df_con_nulos = df.copy()
df_con_nulos.loc[1, 'precio'] = np.nan
df_con_nulos.loc[3, 'stock'] = np.nan

# Ordenar con nulos al final (comportamiento predeterminado)
ordenado_nulos_final = df_con_nulos.sort_values('precio', na_position='last')
print(ordenado_nulos_final)

# Ordenar con nulos al principio
ordenado_nulos_inicio = df_con_nulos.sort_values('precio', na_position='first')
print(ordenado_nulos_inicio)
separator_line()

# Crear un DataFrame con un índice no ordenado
df_indice = df.set_index('producto')
df_indice_desordenado = df_indice.reindex(['Laptop', 'Mouse', 'Tablet', 'Monitor', 'Teclado'])
print(df_indice_desordenado)

# Ordenar por índice (alfabéticamente en este caso)
df_indice_ordenado = df_indice_desordenado.sort_index()
print(df_indice_ordenado)

# Ordenar por índice en orden descendente
df_indice_desc = df_indice_desordenado.sort_index(ascending=False)
print(df_indice_desc)
separator_line()

# Ordenar índice con opciones adicionales
df_indice_con_nulos = df_indice_desordenado.copy()
df_indice_con_nulos.rename(index={'Laptop': None}, inplace=True)

ordenado_indice = df_indice_con_nulos.sort_index(ascending=False, na_position='first')
print(ordenado_indice)
separator_line()

# Crear un DataFrame con índice multinivel
multi_idx = pd.MultiIndex.from_tuples([
    ('Electrónica', 'Laptop'), 
    ('Periféricos', 'Teclado'),
    ('Periféricos', 'Mouse'),
    ('Electrónica', 'Tablet'),
    ('Periféricos', 'Monitor')
], names=['categoría', 'producto'])

df_multi = pd.DataFrame({
    'precio': [1200, 80, 45, 350, 250],
    'stock': [15, 120, 150, 40, 30]
}, index=multi_idx)

print(df_multi)

# Ordenar por el primer nivel del índice
df_multi_ordenado = df_multi.sort_index(level=0)
print(df_multi_ordenado)

# Ordenar por el segundo nivel del índice
df_multi_ordenado_nivel2 = df_multi.sort_index(level=1)
print(df_multi_ordenado_nivel2)

# Ordenar por ambos niveles con diferentes órdenes
df_multi_ordenado_mixto = df_multi.sort_index(level=[0, 1], ascending=[True, False])
print(df_multi_ordenado_mixto)
separator_line()

# Ordenación estable (predeterminada)
df_ordenado_estable = df.sort_values('stock', kind='mergesort')  # algoritmo estable

# Ordenación potencialmente inestable pero más rápida
df_ordenado_rapido = df.sort_values('stock', kind='quicksort')  # algoritmo inestable
separator_line()

# Ordenar creando una copia (comportamiento predeterminado)
df_ordenado = df.sort_values('precio')
print(id(df) != id(df_ordenado))  # True, son objetos diferentes

# Ordenar modificando el DataFrame original
df.sort_values('precio', inplace=True)
print(df)  # El DataFrame original ahora está ordenado
separator_line()

# Productos más caros
productos_mas_caros = df.sort_values('precio', ascending=False).head(2)
print(productos_mas_caros)

# Productos con menor stock
productos_menor_stock = df.sort_values('stock').head(2)
print(productos_menor_stock)
separator_line()

# Ordenar por fecha de ingreso
df_cronologico = df.sort_values('fecha_ingreso')
print(df_cronologico)

# Calcular stock acumulado a lo largo del tiempo
df_cronologico['stock_acumulado'] = df_cronologico['stock'].cumsum()
print(df_cronologico[['fecha_ingreso', 'producto', 'stock', 'stock_acumulado']])
separator_line()

# Ordenar para un gráfico de barras (de mayor a menor)
df_para_grafico = df.sort_values('precio', ascending=False)
# El código para generar el gráfico sería:
# df_para_grafico.plot(x='producto', y='precio', kind='bar')

separator_line()

# Agrupar visualmente por categoría ordenando
df['categoria'] = ['Periféricos', 'Periféricos', 'Periféricos', 
                   'Computación', 'Móviles']
df_agrupado = df.sort_values('categoria')
print(df_agrupado)
separator_line()

# Crear un DataFrame grande
n = 1000000
df_grande = pd.DataFrame({
    'A': np.random.randint(0, 1000, size=n),
    'B': np.random.randint(0, 1000, size=n),
    'C': np.random.randint(0, 1000, size=n)
})

# Comparar rendimiento de diferentes algoritmos
# %timeit -n 1 -r 1 df_grande.sort_values('A', kind='quicksort')
# %timeit -n 1 -r 1 df_grande.sort_values('A', kind='mergesort')
# %timeit -n 1 -r 1 df_grande.sort_values('A', kind='heapsort')
separator_line()

# Creamos un DataFrame con datos de estudiantes
datos = {
    'estudiante': ['Ana', 'Carlos', 'Elena', 'David', 'Beatriz', 'Fernando'],
    'matemáticas': [85, 92, 78, 92, 88, 73],
    'ciencias': [90, 85, 95, 89, 92, 70],
    'literatura': [92, 78, 96, 85, 91, 84]
}

df = pd.DataFrame(datos)
print(df)
separator_line()

# Ranking de calificaciones de matemáticas
ranking_matematicas = df['matemáticas'].rank()
print(ranking_matematicas)
separator_line()

# Ranking descendente (el valor más alto recibe el rango 1)
ranking_desc = df['matemáticas'].rank(ascending=False)
print(ranking_desc)
separator_line()

# Observamos que Carlos y David tienen la misma calificación en matemáticas (92)
print(df[['estudiante', 'matemáticas']])

# Diferentes métodos para manejar empates
print("\nMétodo 'average' (predeterminado):")
print(df['matemáticas'].rank(method='average'))

print("\nMétodo 'min':")
print(df['matemáticas'].rank(method='min'))

print("\nMétodo 'max':")
print(df['matemáticas'].rank(method='max'))

print("\nMétodo 'first':")
print(df['matemáticas'].rank(method='first'))

print("\nMétodo 'dense':")
print(df['matemáticas'].rank(method='dense'))
separator_line()

# Ranking de todas las calificaciones
rankings = df[['matemáticas', 'ciencias', 'literatura']].rank(ascending=False)
df_con_rankings = pd.concat([df, rankings.add_suffix('_ranking')], axis=1)
print(df_con_rankings)
separator_line()

# Ranking de materias para cada estudiante
ranking_por_estudiante = df[['matemáticas', 'ciencias', 'literatura']].rank(axis=1, ascending=False)
ranking_por_estudiante.columns = ['rank_mat', 'rank_ciencias', 'rank_lit']
resultado = pd.concat([df, ranking_por_estudiante], axis=1)
print(resultado)
separator_line()

# Convertir rangos a percentiles
n = len(df)
df['percentil_matematicas'] = df['matemáticas'].rank(pct=True) * 100
print(df[['estudiante', 'matemáticas', 'percentil_matematicas']])
separator_line()

# Identificar estudiantes en el 20% superior o inferior
df['top_matematicas'] = df['matemáticas'].rank(pct=True) >= 0.8
df['bottom_matematicas'] = df['matemáticas'].rank(pct=True) <= 0.2
print(df[['estudiante', 'matemáticas', 'top_matematicas', 'bottom_matematicas']])
separator_line()

# Normalización basada en rangos (0-1)
df['matematicas_norm'] = (df['matemáticas'].rank() - 1) / (len(df) - 1)
print(df[['estudiante', 'matemáticas', 'matematicas_norm']])
separator_line()

# Obtener los 3 estudiantes con mejores calificaciones en matemáticas
mejores_matematicas = df.nlargest(3, 'matemáticas')
print(mejores_matematicas)
separator_line()

# Obtener los 3 mejores estudiantes considerando primero matemáticas y luego ciencias
mejores_estudiantes = df.nlargest(3, ['matemáticas', 'ciencias'])
print(mejores_estudiantes)
separator_line()

# Obtener los 2 estudiantes con calificaciones más bajas en literatura
peores_literatura = df.nsmallest(2, 'literatura')
print(peores_literatura)
separator_line()

# Obtener los 2 estudiantes con peores calificaciones considerando primero ciencias y luego matemáticas
estudiantes_apoyo = df.nsmallest(2, ['ciencias', 'matemáticas'])
print(estudiantes_apoyo)
separator_line()

# Los 3 valores más altos de matemáticas
top3_matematicas = df['matemáticas'].nlargest(3)
print(top3_matematicas)

# Los 2 valores más bajos de ciencias
bottom2_ciencias = df['ciencias'].nsmallest(2)
print(bottom2_ciencias)
separator_line()

# Crear un DataFrame grande
n = 100000
df_grande = pd.DataFrame({
    'A': np.random.randint(0, 10000, size=n),
    'B': np.random.randint(0, 10000, size=n)
})

# Comparación de rendimiento
# %timeit -n 1 -r 1 df_grande.nlargest(10, 'A')
# %timeit -n 1 -r 1 df_grande.sort_values('A', ascending=False).head(10)
separator_line()

# Calcular promedio por estudiante
df['promedio'] = df[['matemáticas', 'ciencias', 'literatura']].mean(axis=1)

# Top 3 estudiantes por promedio
mejores_promedios = df.nlargest(3, 'promedio')
print(mejores_promedios[['estudiante', 'promedio']])
separator_line()

# Identificar las mayores diferencias entre matemáticas y literatura
df['diferencia'] = abs(df['matemáticas'] - df['literatura'])
mayores_diferencias = df.nlargest(2, 'diferencia')
print(mayores_diferencias[['estudiante', 'matemáticas', 'literatura', 'diferencia']])
separator_line()

# Crear un informe con los mejores y peores resultados
def generar_informe(dataframe, columna):
    mejores = dataframe.nlargest(2, columna)[['estudiante', columna]]
    peores = dataframe.nsmallest(2, columna)[['estudiante', columna]]
    return pd.concat([
        pd.DataFrame({'Categoría': ['Mejores desempeños']}),
        mejores,
        pd.DataFrame({'Categoría': ['Peores desempeños']}),
        peores
    ])

informe_matematicas = generar_informe(df, 'matemáticas')
print(informe_matematicas)
separator_line()

# Simular importancia de características
importancia = pd.Series({
    'edad': 0.15,
    'horas_estudio': 0.45,
    'asistencia': 0.25,
    'participacion': 0.08,
    'tareas_completadas': 0.32,
    'examenes_previos': 0.38
})

# Seleccionar las 3 características más importantes
caracteristicas_principales = importancia.nlargest(3)
print("Características más importantes para el modelo:")
print(caracteristicas_principales)
separator_line()

# Calcular el ranking general de estudiantes
df['ranking_general'] = df[['matemáticas', 'ciencias', 'literatura']].mean(axis=1).rank(ascending=False)

# Seleccionar los 3 mejores estudiantes según el ranking
top_estudiantes = df.nsmallest(3, 'ranking_general')
print(top_estudiantes[['estudiante', 'matemáticas', 'ciencias', 'literatura', 'ranking_general']])
separator_line()
