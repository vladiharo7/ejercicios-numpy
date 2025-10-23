'''
Tienes un conjunto de datos de temperaturas diarias (en grados Celsius) registradas durante un mes en tres ciudades diferentes. Utilizando las funciones estadísticas de NumPy, debes:

Calcular la temperatura media para cada ciudad.
Identificar la temperatura máxima y mínima registrada en cada ciudad.
Calcular la mediana de temperaturas para cada ciudad.
Determinar el rango intercuartílico (IQR) de temperaturas para cada ciudad.
Identificar los días con temperaturas atípicas (outliers) en cada ciudad, definiendo como outliers aquellos valores que están fuera del rango [Q1 - 1.5IQR, Q3 + 1.5IQR].
Los datos de temperatura están organizados en un array NumPy donde cada fila representa una ciudad y cada columna representa un día del mes:

import numpy as np

temperaturas = np.array([
    [25, 28, 30, 32, 29, 27, 26, 25, 24, 28, 31, 30, 29, 28, 27, 29, 30, 31, 32, 33, 34, 31, 29, 28, 27, 26, 25, 24, 25, 26],  # Ciudad A
    [18, 17, 19, 20, 21, 20, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20, 21, 22, 21, 20, 19, 18, 17, 16, 15, 14, 15, 16, 17, 18],  # Ciudad B
    [31, 32, 33, 34, 35, 36, 35, 34, 33, 32, 31, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 38, 36, 34, 32, 30, 31, 32, 33]   # Ciudad C
])
Para cada punto, debes mostrar los resultados de manera clara y ordenada.
'''
import numpy as np

temperaturas = np.array([
    [25, 28, 30, 32, 29, 27, 26, 25, 24, 28, 31, 30, 29, 28, 27, 29, 30, 31, 32, 33, 34, 31, 29, 28, 27, 26, 25, 24, 25, 26],  # Ciudad A
    [18, 17, 19, 20, 21, 20, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20, 21, 22, 21, 20, 19, 18, 17, 16, 15, 14, 15, 16, 17, 18],  # Ciudad B
    [31, 32, 33, 34, 35, 36, 35, 34, 33, 32, 31, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 38, 36, 34, 32, 30, 31, 32, 33]   # Ciudad C
])

def estadisticas_temperaturas(temperaturas):
    ciudades = ['Ciudad A', 'Ciudad B', 'Ciudad C']

    for i, ciudad in enumerate(ciudades):
        datos_ciudad = temperaturas[i]

        # Temperatura media
        media = np.mean(datos_ciudad)

        # Temperatura máxima y mínima
        max_temp = np.max(datos_ciudad)
        min_temp = np.min(datos_ciudad)

        # Mediana
        mediana = np.median(datos_ciudad)

        # Rango intercuartílico (IQR)
        Q1 = np.percentile(datos_ciudad, 25)
        Q3 = np.percentile(datos_ciudad, 75)
        IQR = Q3 - Q1

        # Identificación de outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = datos_ciudad[(datos_ciudad < lower_bound) | (datos_ciudad > upper_bound)]

        # Mostrar resultados
        print(f'Estadísticas para {ciudad}:')
        print(f'  Temperatura media: {media:.2f} °C')
        print(f'  Temperatura máxima: {max_temp} °C')
        print(f'  Temperatura mínima: {min_temp} °C')
        print(f'  Mediana: {mediana} °C')
        print(f'  Rango intercuartílico (IQR): {IQR:.2f} °C')
        if outliers.size > 0:
            print(f'  Outliers: {outliers}')
        else:
            print('  No se encontraron outliers.')
        print('')

estadisticas_temperaturas(temperaturas)
