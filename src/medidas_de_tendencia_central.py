import numpy as np

def print_separator():
    print("----------------------------------------------------------")

# Crear un array de ejemplo
datos = np.array([12, 15, 21, 18, 9, 24, 17, 22])

# Calcular la media
media = np.mean(datos)
print(f"Media: {media}")  # Output: Media: 17.25

print_separator()

# Matriz 2D que representa calificaciones de estudiantes en diferentes asignaturas
calificaciones = np.array([
    [85, 90, 78, 92],  # Estudiante 1
    [76, 88, 95, 87],  # Estudiante 2
    [91, 82, 89, 94]   # Estudiante 3
])

# Media por estudiante (a lo largo del eje 1)
media_estudiantes = np.mean(calificaciones, axis=1)
print("Media por estudiante:")
print(media_estudiantes)  # [86.25, 86.5, 89.0]

# Media por asignatura (a lo largo del eje 0)
media_asignaturas = np.mean(calificaciones, axis=0)
print("Media por asignatura:")
print(media_asignaturas)  # [84.0, 86.67, 87.33, 91.0]

print_separator()

# Array con un valor atípico
datos_con_outlier = np.array([12, 15, 21, 18, 9, 24, 17, 22, 150])

# Comparación entre media y mediana
media = np.mean(datos_con_outlier)
mediana = np.median(datos_con_outlier)

print(f"Media: {media}")    # Output: Media: 32.0
print(f"Mediana: {mediana}")  # Output: Mediana: 18.0

# Mediana por estudiante
mediana_estudiantes = np.median(calificaciones, axis=1)
print("Mediana por estudiante:")
print(mediana_estudiantes)  # [87.5, 87.5, 90.0]

# Mediana por asignatura
mediana_asignaturas = np.median(calificaciones, axis=0)
print("Mediana por asignatura:")
print(mediana_asignaturas)  # [85.0, 88.0, 89.0, 92.0]

print_separator()

datos = np.array([12, 15, 21, 18, 9, 24, 17, 22, 30, 25, 11, 14])

# Calcular diferentes percentiles
p25 = np.percentile(datos, 25)  # Primer cuartil (Q1)
p50 = np.percentile(datos, 50)  # Mediana o segundo cuartil (Q2)
p75 = np.percentile(datos, 75)  # Tercer cuartil (Q3)
p90 = np.percentile(datos, 90)  # Percentil 90

print(f"Percentil 25: {p25}")  # Output: Percentil 25: 12.75
print(f"Percentil 50 (mediana): {p50}")  # Output: Percentil 50 (mediana): 17.5
print(f"Percentil 75: {p75}")  # Output: Percentil 75: 23.5
print(f"Percentil 90: {p90}")  # Output: Percentil 90: 27.5

print_separator()

# Calcular el rango intercuartílico (IQR)
q1 = np.percentile(datos, 25)
q3 = np.percentile(datos, 75)
iqr = q3 - q1

# Definir límites para detectar outliers
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

print(f"Rango intercuartílico (IQR): {iqr}")
print(f"Límite inferior para outliers: {limite_inferior}")
print(f"Límite superior para outliers: {limite_superior}")

# Identificar outliers
outliers = datos[(datos < limite_inferior) | (datos > limite_superior)]
print(f"Outliers: {outliers}")

print_separator()


# Calcular el percentil 75 para cada asignatura
p75_asignaturas = np.percentile(calificaciones, 75, axis=0)
print("Percentil 75 por asignatura:")
print(p75_asignaturas)  # [88.0, 89.0, 92.0, 93.0]

# Calcular múltiples percentiles de una vez
percentiles = np.percentile(datos, [25, 50, 75, 90])
print("Múltiples percentiles:")
print(percentiles)  # [12.75, 17.5, 23.5, 27.5]

print_separator()


# Equivalente a percentile(datos, 25)
q25 = np.quantile(datos, 0.25)

# Equivalente a percentile(datos, [25, 50, 75])
quantiles = np.quantile(datos, [0.25, 0.5, 0.75])

print(f"Cuantil 0.25: {q25}")  # Output: Cuantil 0.25: 12.75
print("Múltiples cuantiles:")
print(quantiles)  # [12.75, 17.5, 23.5]

print_separator()

# análisis de datos reales

# Temperaturas diarias (°C) durante un mes
temperaturas = np.array([
    22.5, 25.1, 23.4, 21.9, 20.8, 22.0, 23.5, 24.1, 26.2, 27.5,
    28.3, 29.2, 30.1, 29.8, 28.5, 27.9, 26.3, 25.2, 24.0, 23.1,
    22.8, 21.5, 20.9, 19.8, 18.5, 17.9, 19.2, 20.5, 21.8, 22.6
])

# Calcular medidas de tendencia central
media_temp = np.mean(temperaturas)
mediana_temp = np.median(temperaturas)
temp_min = np.min(temperaturas)
temp_max = np.max(temperaturas)
percentiles = np.percentile(temperaturas, [25, 50, 75])

print(f"Estadísticas de temperatura del mes:")
print(f"Media: {media_temp:.1f}°C")
print(f"Mediana: {mediana_temp:.1f}°C")
print(f"Temperatura mínima: {temp_min:.1f}°C")
print(f"Temperatura máxima: {temp_max:.1f}°C")
print(f"Primer cuartil (Q1): {percentiles[0]:.1f}°C")
print(f"Segundo cuartil (Q2): {percentiles[1]:.1f}°C")
print(f"Tercer cuartil (Q3): {percentiles[2]:.1f}°C")

print_separator()

# Desviación estándar con std()
# Dos conjuntos de datos con la misma media pero diferente dispersión
datos_a = np.array([10, 11, 9, 12, 8, 10, 10])
datos_b = np.array([2, 18, 5, 15, 10, 0, 20])

# Ambos tienen la misma media
print(f"Media de datos_a: {np.mean(datos_a)}")  # 10.0
print(f"Media de datos_b: {np.mean(datos_b)}")  # 10.0

# Pero diferente desviación estándar
std_a = np.std(datos_a)
std_b = np.std(datos_b)

print(f"Desviación estándar de datos_a: {std_a:.2f}")  # 1.29
print(f"Desviación estándar de datos_b: {std_b:.2f}")  # 7.48

print_separator()
# Desviación estándar poblacional (por defecto)
std_poblacional = np.std(datos_a)

# Desviación estándar muestral
std_muestral = np.std(datos_a, ddof=1)

print(f"Desviación estándar poblacional: {std_poblacional:.4f}")  # 1.2910
print(f"Desviación estándar muestral: {std_muestral:.4f}")       # 1.3973

print_separator()

# Matriz de datos de ventas mensuales por región
ventas = np.array([
    [120, 105, 130, 125],  # Región Norte
    [145, 140, 150, 160],  # Región Sur
    [100, 110, 105, 95]    # Región Este
])

# Desviación estándar por región (filas)
std_regiones = np.std(ventas, axis=1)
print("Desviación estándar por región:")
print(std_regiones)  # [9.35, 7.36, 5.59]

# Desviación estándar por mes (columnas)
std_meses = np.std(ventas, axis=0)
print("Desviación estándar por mes:")
print(std_meses)  # [18.33, 15.28, 18.33, 26.25]

print_separator()

# Calcular la varianza
var_a = np.var(datos_a)
var_b = np.var(datos_b)

print(f"Varianza de datos_a: {var_a:.2f}")  # 1.67
print(f"Varianza de datos_b: {var_b:.2f}")  # 56.00

# Verificar relación entre varianza y desviación estándar
print(f"Raíz cuadrada de la varianza de datos_a: {np.sqrt(var_a):.4f}")  # 1.2910
print(f"Desviación estándar de datos_a: {std_a:.4f}")                   # 1.2910

print_separator()

# Varianza poblacional vs muestral
var_poblacional = np.var(datos_a)
var_muestral = np.var(datos_a, ddof=1)

print(f"Varianza poblacional: {var_poblacional:.4f}")  # 1.6667
print(f"Varianza muestral: {var_muestral:.4f}")       # 1.9524
print_separator()

# Datos en diferentes escalas
temperaturas_c = np.array([22, 24, 21, 25, 23])  # Celsius
precios = np.array([1200, 1350, 1100, 1400, 1250])  # Euros

# Calcular coeficientes de variación
cv_temp = (np.std(temperaturas_c) / np.mean(temperaturas_c)) * 100
cv_precios = (np.std(precios) / np.mean(precios)) * 100

print(f"CV temperaturas: {cv_temp:.2f}%")  # 6.52%
print(f"CV precios: {cv_precios:.2f}%")    # 9.13%

print_separator()

# Datos de altura (cm) y peso (kg)
altura = np.array([165, 170, 182, 175, 159, 172, 168])
peso = np.array([62, 67, 81, 73, 55, 69, 63])

# Calcular matriz de covarianza
matriz_cov = np.cov(altura, peso)
print("Matriz de covarianza:")
print(matriz_cov)

# La covarianza entre altura y peso está en las posiciones [0,1] y [1,0]
cov_altura_peso = matriz_cov[0, 1]
print(f"Covarianza entre altura y peso: {cov_altura_peso:.2f}")  # 58.90

print_separator()

# Calcular coeficiente de correlación
corr = np.corrcoef(altura, peso)
print("Matriz de correlación:")
print(corr)

# El coeficiente de correlación entre altura y peso
corr_altura_peso = corr[0, 1]
print(f"Correlación entre altura y peso: {corr_altura_peso:.4f}")  # 0.9838


print_separator()

# análisis de datos financieros

# Rendimientos mensuales (%) de tres activos durante un año
activo_a = np.array([1.2, 0.8, 1.5, -0.5, 1.3, 1.8, 0.9, 1.1, -0.2, 1.4, 1.6, 0.7])
activo_b = np.array([2.1, 1.9, 2.3, 1.8, -0.5, 2.5, 2.0, 1.5, 1.8, 2.2, -0.8, 2.4])
activo_c = np.array([0.5, 0.6, 0.4, 0.7, 0.3, 0.6, 0.5, 0.4, 0.6, 0.5, 0.7, 0.4])

# Rendimiento promedio (retorno esperado)
retorno_a = np.mean(activo_a)
retorno_b = np.mean(activo_b)
retorno_c = np.mean(activo_c)

# Riesgo (desviación estándar)
riesgo_a = np.std(activo_a, ddof=1)
riesgo_b = np.std(activo_b, ddof=1)
riesgo_c = np.std(activo_c, ddof=1)

# Coeficiente de variación (riesgo por unidad de retorno)
cv_a = (riesgo_a / retorno_a) * 100
cv_b = (riesgo_b / retorno_b) * 100
cv_c = (riesgo_c / retorno_c) * 100

print("Análisis de activos financieros:")
print(f"Activo A - Retorno: {retorno_a:.2f}%, Riesgo: {riesgo_a:.2f}%, CV: {cv_a:.2f}")
print(f"Activo B - Retorno: {retorno_b:.2f}%, Riesgo: {riesgo_b:.2f}%, CV: {cv_b:.2f}")
print(f"Activo C - Retorno: {retorno_c:.2f}%, Riesgo: {riesgo_c:.2f}%, CV: {cv_c:.2f}")

# Matriz de correlación entre los activos
activos = np.vstack([activo_a, activo_b, activo_c])
matriz_corr = np.corrcoef(activos)
print("\nMatriz de correlación entre activos:")
print(matriz_corr)

print_separator()

import matplotlib.pyplot as plt

# Datos para visualizar
datos = np.array([15, 17, 12, 20, 16, 15, 18, 14, 19, 13, 16, 17])
media = np.mean(datos)
std = np.std(datos)

# Crear histograma con líneas para la media y desviaciones estándar
plt.figure(figsize=(10, 6))
plt.hist(datos, bins=6, alpha=0.7, color='skyblue')
plt.axvline(media, color='red', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')
plt.axvline(media + std, color='green', linestyle='dotted', linewidth=2, label=f'Media + 1 Std: {media + std:.2f}')
plt.axvline(media - std, color='green', linestyle='dotted', linewidth=2, label=f'Media - 1 Std: {media - std:.2f}')
plt.title('Distribución de datos con media y desviación estándar')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print_separator()

# Array unidimensional
ventas_semanales = np.array([1200, 1450, 1300, 1390, 1600, 950, 1100])

# Suma total
total_ventas = np.sum(ventas_semanales)
print(f"Total de ventas: ${total_ventas}")  # Output: Total de ventas: $8990

print_separator()

# Matriz de ventas por producto (filas) y región (columnas)
ventas_matriz = np.array([
    [120, 85, 90, 110],    # Producto A
    [95, 100, 85, 120],    # Producto B
    [110, 90, 100, 105]    # Producto C
])

# Suma por producto (a lo largo del eje 1)
ventas_por_producto = np.sum(ventas_matriz, axis=1)
print("Ventas totales por producto:")
print(ventas_por_producto)  # [405, 400, 405]

# Suma por región (a lo largo del eje 0)
ventas_por_region = np.sum(ventas_matriz, axis=0)
print("Ventas totales por región:")
print(ventas_por_region)    # [325, 275, 275, 335]

print_separator()

# Mantener dimensiones al sumar
suma_con_dims = np.sum(ventas_matriz, axis=0, keepdims=True)
print("Suma manteniendo dimensiones:")
print(suma_con_dims)  # [[325, 275, 275, 335]]

# Útil para calcular porcentajes respecto al total
porcentajes = ventas_matriz / suma_con_dims * 100
print("Porcentaje de ventas por producto y región:")
print(porcentajes)

print_separator()

# Calcular factorial con prod()
n = 5
numeros = np.arange(1, n+1)  # [1, 2, 3, 4, 5]
factorial = np.prod(numeros)
print(f"{n}! = {factorial}")  # Output: 5! = 120

# Calcular crecimiento compuesto
tasas_crecimiento = np.array([1.03, 1.02, 1.05, 1.04, 1.03])  # Tasas anuales (1+r)
crecimiento_total = np.prod(tasas_crecimiento) - 1
print(f"Crecimiento total: {crecimiento_total:.2%}")  # Output: Crecimiento total: 18.16%


print_separator()

# Matriz de factores de crecimiento por trimestre (filas) y producto (columnas)
factores_crecimiento = np.array([
    [1.02, 1.03, 1.01],  # Q1
    [1.01, 1.02, 1.03],  # Q2
    [1.03, 1.01, 1.02],  # Q3
    [1.02, 1.04, 1.01]   # Q4
])

# Crecimiento anual por producto
crecimiento_anual = np.prod(factores_crecimiento, axis=0) - 1
print("Crecimiento anual por producto:")
#print(f"{crecimiento_anual * 100:.2f}%")  # [8.24%, 10.36%, 7.18%]
print([f"{valor:.2f}%" for valor in (crecimiento_anual * 100)])  # [8.24%, 10.36%, 7.18%]

# Crecimiento acumulado por trimestre
crecimiento_trimestral = np.prod(factores_crecimiento, axis=1) - 1
print("Crecimiento por trimestre (todos los productos):")
for i, crecimiento in enumerate(crecimiento_trimestral):
    print(f"Q{i+1}: {crecimiento:.2%}")

print_separator()

# Ventas mensuales
ventas_mensuales = np.array([45000, 52000, 49000, 58000, 56000, 60000])

# Ventas acumuladas
ventas_acumuladas = np.cumsum(ventas_mensuales)
print("Ventas acumuladas:")
print(ventas_acumuladas)  # [45000, 97000, 146000, 204000, 260000, 320000]

# Visualizar progreso hacia meta anual
meta_anual = 350000
porcentaje_meta = ventas_acumuladas / meta_anual * 100

print("Progreso hacia meta anual:")
for mes, (ventas, acumulado, porcentaje) in enumerate(zip(ventas_mensuales, 
                                                         ventas_acumuladas, 
                                                         porcentaje_meta), 1):
    print(f"Mes {mes}: ${ventas} | Acumulado: ${acumulado} | {porcentaje:.1f}% de la meta")

print_separator()

# Matriz de datos de lluvia mensual (filas=años, columnas=meses)
lluvia = np.array([
    [45, 35, 60, 40, 30, 10, 5, 10, 25, 50, 70, 80],  # 2020
    [50, 40, 55, 35, 25, 15, 8, 12, 30, 60, 75, 85]   # 2021
])

# Lluvia acumulada por año (a lo largo del eje 1)
lluvia_acumulada = np.cumsum(lluvia, axis=1)
print("Lluvia acumulada por mes (mm):")
print(lluvia_acumulada)

# Comparar patrones de lluvia entre años
for año, datos in enumerate([2020, 2021]):
    print(f"\nLluvia acumulada {datos}:")
    for mes, acumulado in enumerate(lluvia_acumulada[año], 1):
        print(f"Mes {mes}: {acumulado} mm")

print_separator()
# Tasas de crecimiento mensual (como 1 + tasa)
tasas_mensuales = np.array([1.01, 1.02, 0.99, 1.03, 1.01, 1.02])

# Factores de crecimiento acumulados
crecimiento_acumulado = np.cumprod(tasas_mensuales)
print("Factores de crecimiento acumulados:")
print(crecimiento_acumulado)  # [1.01, 1.0302, 1.019898, 1.05049494, 1.06099989, 1.08221989]

# Convertir a porcentajes de crecimiento
porcentaje_crecimiento = (crecimiento_acumulado - 1) * 100
print("Crecimiento acumulado (%):")
for mes, porcentaje in enumerate(porcentaje_crecimiento, 1):
    print(f"Mes {mes}: {porcentaje:.2f}%")


print_separator()

#cálculo del interés compuesto o el valor futuro de inversiones

# Inversión inicial
inversion_inicial = 10000

# Rendimientos mensuales (como 1 + tasa)
rendimientos = np.array([1.012, 1.008, 1.015, 0.995, 1.020, 1.010, 
                         1.005, 1.018, 1.012, 1.008, 1.022, 1.010])

# Valor de la inversión a lo largo del tiempo
valores_inversion = inversion_inicial * np.cumprod(rendimientos)

print("Evolución del valor de la inversión:")
for mes, valor in enumerate(valores_inversion, 1):
    print(f"Mes {mes}: ${valor:.2f}")

# Rendimiento total al final del período
rendimiento_total = (valores_inversion[-1] / inversion_inicial - 1) * 100
print(f"\nRendimiento total anual: {rendimiento_total:.2f}%")

print_separator()

# Datos de ventas diarias durante dos semanas
ventas_diarias = np.array([
    [120, 135, 110, 140, 125, 90, 80],    # Semana 1
    [130, 145, 115, 150, 135, 100, 95]    # Semana 2
])

# Ventas totales por semana
totales_semanales = np.sum(ventas_diarias, axis=1)
print("Ventas totales por semana:")
print(totales_semanales)  # [800, 870]

# Ventas acumuladas por día de la semana
acumulado_por_dia = np.cumsum(ventas_diarias, axis=0)
print("Ventas acumuladas por día:")
print(acumulado_por_dia)

# Calcular el promedio de ventas diarias
promedio_diario = np.mean(ventas_diarias, axis=0)
print("Promedio de ventas por día de la semana:")
print(promedio_diario)

# Calcular el crecimiento porcentual de la semana 2 respecto a la semana 1
crecimiento = (ventas_diarias[1] / ventas_diarias[0] - 1) * 100
print("Crecimiento porcentual por día:")
print([f"{valor:.1f}%" for valor in crecimiento])
print_separator()

# análisis de rendimiento de inversiones

# Rendimientos mensuales (%) para tres estrategias de inversión
estrategia_a = np.array([1.2, 0.8, -0.5, 1.5, 0.9, 1.3, 0.7, 1.1, -0.3, 1.0, 1.4, 0.6])
estrategia_b = np.array([0.5, 0.6, 0.7, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.7, 0.5, 0.6])
estrategia_c = np.array([2.1, 1.5, -1.2, 2.3, -0.8, 1.9, 1.2, -0.5, 1.8, 2.0, -0.7, 1.5])

# Convertir a factores de crecimiento (1 + r/100)
factores_a = 1 + estrategia_a/100
factores_b = 1 + estrategia_b/100
factores_c = 1 + estrategia_c/100

# Calcular rendimiento acumulado
acumulado_a = np.cumprod(factores_a)
acumulado_b = np.cumprod(factores_b)
acumulado_c = np.cumprod(factores_c)

# Rendimiento total anual
rendimiento_total_a = (acumulado_a[-1] - 1) * 100
rendimiento_total_b = (acumulado_b[-1] - 1) * 100
rendimiento_total_c = (acumulado_c[-1] - 1) * 100

print("Rendimiento total anual:")
print(f"Estrategia A: {rendimiento_total_a:.2f}%")
print(f"Estrategia B: {rendimiento_total_b:.2f}%")
print(f"Estrategia C: {rendimiento_total_c:.2f}%")

# Calcular volatilidad (desviación estándar)
volatilidad_a = np.std(estrategia_a)
volatilidad_b = np.std(estrategia_b)
volatilidad_c = np.std(estrategia_c)

print("\nVolatilidad (desviación estándar):")
print(f"Estrategia A: {volatilidad_a:.2f}%")
print(f"Estrategia B: {volatilidad_b:.2f}%")
print(f"Estrategia C: {volatilidad_c:.2f}%")

# Calcular ratio rendimiento/volatilidad
ratio_a = rendimiento_total_a / volatilidad_a
ratio_b = rendimiento_total_b / volatilidad_a
ratio_c = rendimiento_total_c / volatilidad_c

print("\nRatio rendimiento/volatilidad:")
print(f"Estrategia A: {ratio_a:.2f}")
print(f"Estrategia B: {ratio_b:.2f}")
print(f"Estrategia C: {ratio_c:.2f}")


print_separator()

import time

# Crear un array grande
n = 10_000_000
datos_grandes = np.random.rand(n)

# Suma con Python puro
inicio = time.time()
suma_python = sum(datos_grandes)
tiempo_python = time.time() - inicio

# Suma con NumPy
inicio = time.time()
suma_numpy = np.sum(datos_grandes)
tiempo_numpy = time.time() - inicio

print(f"Suma de {n:,} elementos:")
print(f"Python puro: {tiempo_python:.6f} segundos")
print(f"NumPy: {tiempo_numpy:.6f} segundos")
print(f"NumPy es {tiempo_python/tiempo_numpy:.1f}x más rápido")
print_separator()

# Estadísticas por ejes y máscaras condicionales

# Datos de temperatura diaria (°C) para tres ciudades durante una semana
temperaturas = np.array([
    [25, 28, 30, 32, 29, 27, 26],  # Ciudad A
    [20, 22, 21, 23, 24, 19, 18],  # Ciudad B
    [31, 32, 35, 33, 30, 29, 28]   # Ciudad C
])

# Temperatura media por ciudad (a lo largo del eje 1)
temp_media_ciudad = np.mean(temperaturas, axis=1)
print("Temperatura media por ciudad:")
print(temp_media_ciudad)  # [28.14, 21.0, 31.14]

# Temperatura media por día (a lo largo del eje 0)
temp_media_dia = np.mean(temperaturas, axis=0)
print("Temperatura media por día:")
print(temp_media_dia)  # [25.33, 27.33, 28.67, 29.33, 27.67, 25.0, 24.0]

print_separator()

# Temperatura máxima por ciudad
temp_max_ciudad = np.max(temperaturas, axis=1)
print("Temperatura máxima por ciudad:")
print(temp_max_ciudad)  # [32, 24, 35]

# Día con temperatura mínima por ciudad (devuelve el índice)
dia_min_temp = np.argmin(temperaturas, axis=1)
print("Día con temperatura mínima (índice):")
print(dia_min_temp)  # [0, 6, 6]

# Convertir índice a día de la semana
dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
for i, ciudad in enumerate(["A", "B", "C"]):
    print(f"Ciudad {ciudad}: día más frío fue {dias[dia_min_temp[i]]}")

print_separator()

# Datos de ventas por tienda (filas), producto (columnas) y mes (profundidad)
ventas = np.array([
    # Tienda 1
    [[100, 120, 140],    # Producto A (3 meses)
     [80, 85, 90],       # Producto B
     [45, 50, 55]],      # Producto C
    
    # Tienda 2
    [[90, 100, 110],     # Producto A
     [70, 75, 80],       # Producto B
     [40, 45, 50]]       # Producto C
])

# Ventas totales por tienda y producto (suma a lo largo del eje 2 - meses)
total_tienda_producto = np.sum(ventas, axis=2)
print("Ventas totales por tienda y producto:")
print(total_tienda_producto)

# Ventas promedio por producto y mes (promedio a lo largo del eje 0 - tiendas)
promedio_producto_mes = np.mean(ventas, axis=0)
print("\nVentas promedio por producto y mes:")
print(promedio_producto_mes)

print_separator()

# Macaras condicionales

# Datos de temperatura
temperaturas_semana = np.array([25, 28, 30, 32, 29, 27, 26])

# Crear una máscara para días calurosos (>28°C)
dias_calurosos = temperaturas_semana > 28
print("Máscara de días calurosos:")
print(dias_calurosos)  # [False, False, True, True, True, False, False]

# Usar la máscara para filtrar temperaturas
temp_dias_calurosos = temperaturas_semana[dias_calurosos]
print("Temperaturas en días calurosos:")
print(temp_dias_calurosos)  # [30, 32, 29]

# Calcular estadísticas solo para los días calurosos
print(f"Temperatura media en días calurosos: {np.mean(temp_dias_calurosos):.1f}°C")
print(f"Temperatura máxima en días calurosos: {np.max(temp_dias_calurosos)}°C")

print_separator()

# Datos de temperatura y humedad
temperaturas = np.array([25, 28, 30, 32, 29, 27, 26])
humedad = np.array([60, 65, 70, 80, 85, 75, 62])

# Días calurosos Y húmedos (temperatura > 28 Y humedad > 70)
dias_calurosos_humedos = (temperaturas > 28) & (humedad > 70)
print("Días calurosos y húmedos:")
print(dias_calurosos_humedos)  # [False, False, False, True, True, False, False]

# Días calurosos O muy húmedos (temperatura > 28 O humedad > 80)
dias_calurosos_o_muy_humedos = (temperaturas > 28) | (humedad > 80)
print("Días calurosos o muy húmedos:")
print(dias_calurosos_o_muy_humedos)  # [False, False, True, True, True, False, False]

print_separator()

# Datos de ventas mensuales por producto
ventas = np.array([
    [150, 120, 180, 90, 210, 160, 140, 130, 170, 190, 110, 200],  # Producto A
    [80, 75, 95, 85, 100, 90, 70, 65, 85, 95, 60, 110],           # Producto B
    [45, 40, 55, 35, 60, 50, 30, 25, 55, 65, 20, 70]              # Producto C
])

# Crear máscaras para diferentes temporadas
temporada_alta = np.zeros(12, dtype=bool)
temporada_alta[[4, 5, 9, 10, 11]] = True  # Mayo, Junio, Octubre, Noviembre, Diciembre

# Ventas en temporada alta vs. baja por producto
ventas_temp_alta = np.mean(ventas[:, temporada_alta], axis=1)
ventas_temp_baja = np.mean(ventas[:, ~temporada_alta], axis=1)

print("Ventas promedio en temporada alta por producto:")
for i, producto in enumerate(["A", "B", "C"]):
    print(f"Producto {producto}: {ventas_temp_alta[i]:.1f}")

print("\nVentas promedio en temporada baja por producto:")
for i, producto in enumerate(["A", "B", "C"]):
    print(f"Producto {producto}: {ventas_temp_baja[i]:.1f}")

# Calcular incremento porcentual
incremento = (ventas_temp_alta / ventas_temp_baja - 1) * 100
print("\nIncremento en temporada alta vs. baja (%):")
for i, producto in enumerate(["A", "B", "C"]):
    print(f"Producto {producto}: {incremento[i]:.1f}%")


print_separator()


# Datos de calificaciones de estudiantes en diferentes asignaturas
calificaciones = np.array([
    [85, 90, 78, 92, 88],  # Estudiante 1
    [76, 88, 95, 87, 92],  # Estudiante 2
    [91, 82, 89, 94, 60],  # Estudiante 3
    [65, 70, 75, 68, 72]   # Estudiante 4
])

# Contar calificaciones sobresalientes (>90) por estudiante
sobresalientes_por_estudiante = np.sum(calificaciones > 90, axis=1)
print("Número de calificaciones sobresalientes por estudiante:")
print(sobresalientes_por_estudiante)  # [1, 1, 2, 0]

# Contar estudiantes aprobados (>=70) por asignatura
aprobados_por_asignatura = np.sum(calificaciones >= 70, axis=0)
print("Número de estudiantes aprobados por asignatura:")
print(aprobados_por_asignatura)  # [3, 4, 4, 4, 3]

# Promedio de calificaciones aprobadas por asignatura
# Usamos np.where para reemplazar reprobados con NaN y luego nanmean para ignorarlos
from numpy import nan
calificaciones_aprobadas = np.where(calificaciones >= 70, calificaciones, nan)
promedio_aprobados = np.nanmean(calificaciones_aprobadas, axis=0)

print("Promedio de calificaciones aprobadas por asignatura:")
print(promedio_aprobados)


print_separator()

# Datos de tiempo de respuesta de un servidor (ms)
tiempos = np.array([45, 48, 52, 47, 50, 53, 49, 51, 150, 48, 52, 47, 49, 46, 180, 51])

# Calcular estadísticas básicas
media = np.mean(tiempos)
std = np.std(tiempos)

# Identificar outliers (valores a más de 2 desviaciones estándar de la media)
umbral = 2
outliers_mask = np.abs(tiempos - media) > umbral * std
print("Máscara de outliers:")
print(outliers_mask)

# Valores identificados como outliers
outliers = tiempos[outliers_mask]
print("Valores atípicos detectados:")
print(outliers)  # [150, 180]

# Estadísticas sin outliers
tiempos_limpios = tiempos[~outliers_mask]
print(f"Media con outliers: {media:.2f} ms")
print(f"Media sin outliers: {np.mean(tiempos_limpios):.2f} ms")
print(f"Desviación estándar con outliers: {std:.2f} ms")
print(f"Desviación estándar sin outliers: {np.std(tiempos_limpios):.2f} ms")


print_separator()

# Datos de consumo diario de energía (kWh)
consumo = np.array([120, 135, 128, 142, 130, 125, 240, 132, 138, 145, 260, 134])

# Calcular cuartiles
q1 = np.percentile(consumo, 25)
q3 = np.percentile(consumo, 75)
iqr = q3 - q1

# Definir límites para outliers (típicamente 1.5*IQR)
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

# Crear máscara para outliers
outliers_mask = (consumo < limite_inferior) | (consumo > limite_superior)
outliers = consumo[outliers_mask]

print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
print(f"Límite inferior: {limite_inferior}, Límite superior: {limite_superior}")
print(f"Outliers detectados: {outliers}")

# Estadísticas robustas (sin outliers)
consumo_normal = consumo[~outliers_mask]
print(f"Consumo medio normal: {np.mean(consumo_normal):.2f} kWh")
print(f"Desviación estándar normal: {np.std(consumo_normal):.2f} kWh")

# análisis de datos meteorológicos

# Datos de temperatura diaria (°C) para tres estaciones durante un mes
temperaturas = np.array([
    [25, 26, 24, 25, 27, 28, 30, 31, 32, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 29, 28],  # Estación A
    [18, 19, 17, 16, 18, 20, 22, 23, 24, 25, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20, 21, 20, 19],  # Estación B
    [30, 31, 29, 28, 30, 32, 34, 36, 38, 40, 39, 37, 35, 33, 32, 31, 30, 29, 28, 27, 28, 29, 30, 31, 32, 33, 34, 35, 34, 33]   # Estación C
])

# Datos de precipitación diaria (mm)
precipitacion = np.array([
    [0, 0, 5, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 15, 20, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Estación A
    [0, 0, 8, 12, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 18, 25, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Estación B
    [0, 0, 2, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 15, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # Estación C
])

# 1. Identificar días de lluvia
dias_lluvia = precipitacion > 0

# 2. Temperatura media en días de lluvia vs. días secos por estación
for i, estacion in enumerate(["A", "B", "C"]):
    temp_dias_lluvia = temperaturas[i, dias_lluvia[i]]
    temp_dias_secos = temperaturas[i, ~dias_lluvia[i]]
    
    print(f"\nEstación {estacion}:")
    print(f"Temperatura media en días de lluvia: {np.mean(temp_dias_lluvia):.1f}°C")
    print(f"Temperatura media en días secos: {np.mean(temp_dias_secos):.1f}°C")
    print(f"Diferencia: {np.mean(temp_dias_lluvia) - np.mean(temp_dias_secos):.1f}°C")

# 3. Identificar días calurosos por estación (temperatura > percentil 75)
for i, estacion in enumerate(["A", "B", "C"]):
    umbral = np.percentile(temperaturas[i], 75)
    dias_calurosos = temperaturas[i] > umbral
    
    print(f"\nEstación {estacion}:")
    print(f"Umbral para días calurosos (P75): {umbral}°C")
    print(f"Número de días calurosos: {np.sum(dias_calurosos)}")
    print(f"Temperatura media en días calurosos: {np.mean(temperaturas[i, dias_calurosos]):.1f}°C")

# 4. Análisis de correlación entre temperatura y precipitación en días de lluvia
for i, estacion in enumerate(["A", "B", "C"]):
    mask = precipitacion[i] > 0
    if np.sum(mask) > 1:  # Necesitamos al menos 2 puntos para correlación
        temp_lluvia = temperaturas[i, mask]
        precip_lluvia = precipitacion[i, mask]
        corr = np.corrcoef(temp_lluvia, precip_lluvia)[0, 1]
        print(f"\nEstación {estacion} - Correlación temperatura/precipitación: {corr:.2f}")

print_separator()

# Enfoque tradicional con bucle
def normalizar_tradicional(datos, min_val, max_val):
    resultado = np.zeros_like(datos, dtype=float)
    for i in range(len(datos)):
        if datos[i] < min_val:
            resultado[i] = 0
        elif datos[i] > max_val:
            resultado[i] = 1
        else:
            resultado[i] = (datos[i] - min_val) / (max_val - min_val)
    return resultado

# Enfoque con máscaras booleanas
def normalizar_con_mascaras(datos, min_val, max_val):
    resultado = np.zeros_like(datos, dtype=float)
    
    # Crear máscaras para cada condición
    bajo_minimo = datos < min_val
    sobre_maximo = datos > max_val
    en_rango = ~bajo_minimo & ~sobre_maximo
    
    # Aplicar operaciones según las máscaras
    resultado[bajo_minimo] = 0
    resultado[sobre_maximo] = 1
    resultado[en_rango] = (datos[en_rango] - min_val) / (max_val - min_val)
    
    return resultado

# Comparar rendimiento con un array grande
datos_grandes = np.random.normal(50, 15, size=1_000_000)

import time

# Medir tiempo con enfoque tradicional
inicio = time.time()
resultado1 = normalizar_tradicional(datos_grandes, 30, 70)
tiempo1 = time.time() - inicio

# Medir tiempo con máscaras
inicio = time.time()
resultado2 = normalizar_con_mascaras(datos_grandes, 30, 70)
tiempo2 = time.time() - inicio

print(f"Tiempo con bucle: {tiempo1:.4f} segundos")
print(f"Tiempo con máscaras: {tiempo2:.4f} segundos")
print(f"Aceleración: {tiempo1/tiempo2:.1f}x")

