import pandas as pd
import numpy as np

def separator_line():
    print("\n" + "-"*50 + "n")

# Convertir una cadena simple a datetime
fecha = pd.to_datetime('2023-05-15')
print(fecha)

separator_line()

# Convertir una lista de fechas en diferentes formatos
fechas_diversas = ['2023-01-15', '15/02/2023', 'March 10, 2023', '2023.04.20']
fechas_convertidas = pd.to_datetime(fechas_diversas, dayfirst=True, errors='coerce')
print(fechas_convertidas)

separator_line()

# Formato personalizado usando códigos de formato de datetime
fecha_personalizada = pd.to_datetime('15-05-2023 14:30:25', format='%d-%m-%Y %H:%M:%S')
print(fecha_personalizada)

separator_line()

# Lista con una fecha inválida
fechas_con_error = ['2023-01-15', 'no es una fecha', '2023-03-10']

# Por defecto, lanza un error
try:
    pd.to_datetime(fechas_con_error)
except Exception as e:
    print(f"Error: {e}")

# Con errors='coerce', convierte valores inválidos a NaT (Not a Time)
fechas_corregidas = pd.to_datetime(fechas_con_error, errors='coerce')
print(fechas_corregidas)

# Con errors='ignore', deja los valores inválidos sin cambios
fechas_ignoradas = pd.to_datetime(fechas_con_error, errors='ignore')
print(fechas_ignoradas)

separator_line()

# Crear un DataFrame con fechas en formato de texto
df = pd.DataFrame({
    'fecha': ['2023-01-15', '2023-02-20', '2023-03-25'],
    'valor': [100, 150, 200]
})

# Convertir la columna 'fecha' a datetime
df['fecha'] = pd.to_datetime(df['fecha'])
print(df.dtypes)

separator_line()

# Extraer componentes de la fecha
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['día'] = df['fecha'].dt.day
df['día_semana'] = df['fecha'].dt.day_name()

print(df)

separator_line()

# Crear una fecha con zona horaria específica
fecha_con_tz = pd.to_datetime('2023-05-15 14:30:00').tz_localize('Europe/Madrid')
print(fecha_con_tz)

# Convertir a otra zona horaria
fecha_nueva_tz = fecha_con_tz.tz_convert('America/New_York')
print(fecha_nueva_tz)

separator_line()

import pytz

# Listar todas las zonas horarias disponibles
print(len(pytz.all_timezones))  # Muestra el número de zonas horarias disponibles

separator_line()

# Convertir timestamp de Unix (en segundos)
timestamp_unix = 1621008000  # 15 de mayo de 2021
fecha_desde_unix = pd.to_datetime(timestamp_unix, unit='s')
print(fecha_desde_unix)

# Convertir timestamp en milisegundos
timestamp_ms = 1621008000000  # El mismo tiempo en milisegundos
fecha_desde_ms = pd.to_datetime(timestamp_ms, unit='ms')
print(fecha_desde_ms)

separator_line()

# Crear un DataFrame con fechas en varios formatos
datos = {
    'id': range(1, 6),
    'fecha_compra': ['2023-01-15', '20/02/2023', 'March 10, 2023', '2023.04.20', None],
    'monto': [120.50, 300.75, 90.20, 450.00, 200.30]
}

df = pd.DataFrame(datos)
print("DataFrame original:")
print(df)

# Convertir la columna de fechas, manejando valores faltantes
df['fecha_compra'] = pd.to_datetime(df['fecha_compra'], errors='coerce')

# Filtrar compras de un mes específico
compras_marzo = df[df['fecha_compra'].dt.month == 3]
print("\nCompras de marzo:")
print(compras_marzo)

# Calcular días transcurridos desde la compra
hoy = pd.to_datetime('2023-05-15')
df['dias_desde_compra'] = (hoy - df['fecha_compra']).dt.days

print("\nDías transcurridos desde cada compra:")
print(df)

separator_line()

# DataFrame con fechas en formatos mixtos y anotaciones
datos_mixtos = {
    'evento': ['Reunión', 'Conferencia', 'Taller', 'Seminario'],
    'fecha': ['2023-05-15 (confirmado)', '20/06/2023 - pendiente', 
              'Julio 10, 2023 [virtual]', '2023.08.20 presencial']
}

df_eventos = pd.DataFrame(datos_mixtos)
print("Datos originales:")
print(df_eventos)

# Función para extraer la parte de fecha de cada cadena
def extraer_fecha(texto):
    # Patrones comunes para extraer
    patrones = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
        r'[A-Za-z]+ \d{1,2}, \d{4}',  # Month DD, YYYY
        r'\d{4}\.\d{2}\.\d{2}'  # YYYY.MM.DD
    ]
    
    import re
    for patron in patrones:
        match = re.search(patron, texto)
        if match:
            return match.group(0)
    return None

# Aplicar la función y convertir a datetime
df_eventos['fecha_limpia'] = df_eventos['fecha'].apply(extraer_fecha)
df_eventos['fecha_dt'] = pd.to_datetime(df_eventos['fecha_limpia'], errors='coerce')

print("\nDatos procesados:")
print(df_eventos)

separator_line()

# Crear un rango de fechas básico (diario por defecto)
fechas_diarias = pd.date_range(start='2023-01-01', end='2023-01-10')
print(fechas_diarias)

# Rango de fechas mensuales
fechas_mensuales = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
print(fechas_mensuales)

# Rango de fechas semanales (cada lunes)
fechas_semanales = pd.date_range(start='2023-01-01', end='2023-03-01', freq='W-MON')
print(fechas_semanales)

# Rango de fechas con horas (cada 6 horas)
fechas_horas = pd.date_range(start='2023-01-01', periods=5, freq='6H')
print(fechas_horas)

separator_line()

# Especificando start y periods (sin end)
fechas_10_dias = pd.date_range(start='2023-01-01', periods=10)
print(fechas_10_dias)

# Especificando end y periods (sin start)
fechas_hacia_atras = pd.date_range(end='2023-01-10', periods=5)
print(fechas_hacia_atras)

separator_line()

# Cada 2 semanas
quincenas = pd.date_range(start='2023-01-01', periods=6, freq='2W')
print(quincenas)

# Cada 3 meses (trimestral)
trimestres = pd.date_range(start='2023-01-01', periods=4, freq='3M')
print(trimestres)

separator_line()
# Solo días hábiles (lunes a viernes)
dias_habiles = pd.date_range(start='2023-01-01', end='2023-01-15', freq='B')
print(dias_habiles)

separator_line()

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pandas as pd

# Crear calendario de festivos
calendario = USFederalHolidayCalendar()
festivos = calendario.holidays(start='2023-01-01', end='2023-12-31')

# Crear frecuencia personalizada de días hábiles excluyendo festivos
dias_habiles = CustomBusinessDay(holidays=festivos)

# Generar rango de fechas
dias_habiles_sin_festivos = pd.date_range(start='2023-01-01', end='2023-01-31', freq=dias_habiles)

print(dias_habiles_sin_festivos)

separator_line()
# Crear un rango de períodos mensuales
periodos_mensuales = pd.period_range(start='2023-01', end='2023-12', freq='M')
print(periodos_mensuales)
separator_line()

# Trimestres fiscales (comenzando en abril)
trimestres_fiscales = pd.period_range(start='2023Q1', periods=4, freq='Q-MAR')
print(trimestres_fiscales)

# Años académicos (septiembre a agosto)
años_academicos = pd.period_range(start='2023-09', periods=3, freq='A-AUG')
print(años_academicos)
separator_line()

# Comparación de ambos tipos
fechas = pd.date_range('2023-01-01', periods=3, freq='M')
periodos = pd.period_range('2023-01', periods=3, freq='M')

print("DatetimeIndex:")
print(fechas)
print("\nPeriodIndex:")
print(periodos)

separator_line()

# Crear una serie temporal con datos aleatorios
fechas = pd.date_range('2023-01-01', periods=100, freq='D')
valores = np.random.randn(100).cumsum()  # Camino aleatorio

serie_temporal = pd.Series(valores, index=fechas)
print(serie_temporal.head())

# Visualizar la serie
import matplotlib.pyplot as plt
serie_temporal.plot(figsize=(10, 6), title='Serie temporal aleatoria')
plt.tight_layout()
plt.show()

separator_line()

# Crear un DataFrame con datos de ventas diarias
fechas = pd.date_range('2023-01-01', periods=90, freq='D')
datos = {
    'ventas': np.random.randint(100, 500, size=90),
    'visitas': np.random.randint(500, 1000, size=90)
}

df_ventas = pd.DataFrame(datos, index=fechas)
print(df_ventas.head())

# Resample a nivel semanal (suma)
ventas_semanales = df_ventas.resample('W').sum()
print("\nVentas semanales:")
print(ventas_semanales.head())

separator_line()

# Generar fechas de negociación (días hábiles)
dias_negociacion = pd.date_range(
    start='2023-01-01',
    end='2023-12-31',
    freq='B'  # Días hábiles (lunes a viernes)
)

# Simular precios de acciones
precios = 100 + np.cumsum(np.random.normal(0, 1, len(dias_negociacion)))
df_acciones = pd.DataFrame({'precio': precios}, index=dias_negociacion)

# Calcular rendimientos diarios
df_acciones['rendimiento'] = df_acciones['precio'].pct_change() * 100

print(df_acciones.head())

separator_line()

# Crear un calendario de eventos mensuales
meses = pd.period_range('2023-01', '2023-12', freq='M')
eventos = [f"Evento {i+1}" for i in range(len(meses))]

calendario_eventos = pd.Series(eventos, index=meses)
print(calendario_eventos)

# Convertir períodos a timestamps (para el último día del mes)
fechas_eventos = calendario_eventos.index.to_timestamp(how='end')
print("\nFechas de los eventos (último día del mes):")
print(fechas_eventos)

separator_line()

# Convertir DatetimeIndex a PeriodIndex
fechas = pd.date_range('2023-01-01', periods=5, freq='D')
periodos = fechas.to_period('D')
print(f"Original (DatetimeIndex): {fechas}")
print(f"Convertido (PeriodIndex): {periodos}")

# Convertir PeriodIndex a DatetimeIndex
periodos_mes = pd.period_range('2023-01', periods=3, freq='M')
fechas_inicio = periodos_mes.to_timestamp(how='start')  # Primer día del período
fechas_fin = periodos_mes.to_timestamp(how='end')      # Último día del período

print(f"\nPeríodos originales: {periodos_mes}")
print(f"Fechas de inicio: {fechas_inicio}")
print(f"Fechas de fin: {fechas_fin}")


separator_line()

# Días específicos de la semana (lunes y jueves)
dias_especificos = pd.date_range(
    start='2023-01-01',
    end='2023-01-31',
    freq='WOM-1THU'  # Primer jueves del mes
)
print(dias_especificos)

# Último día hábil de cada mes
ultimo_dia_habil = pd.date_range(
    start='2023-01-01',
    end='2023-12-31',
    freq='BM'  # Business Month End
)
print(ultimo_dia_habil)

separator_line()

import matplotlib.pyplot as plt

# Crear una serie temporal diaria
fechas = pd.date_range('2023-01-01', '2023-01-31', freq='D')
valores = np.random.normal(0, 1, size=len(fechas)).cumsum()
serie_diaria = pd.Series(valores, index=fechas)

print("Serie original (diaria):")
print(serie_diaria.head())

separator_line()
# Downsampling: de diario a semanal
serie_semanal = serie_diaria.resample('W').mean()
print("\nSerie remuestreada a semanal (promedio):")
print(serie_semanal)

# Visualizar el efecto del downsampling
fig, ax = plt.subplots(figsize=(10, 6))
serie_diaria.plot(ax=ax, marker='.', label='Datos diarios')
serie_semanal.plot(ax=ax, marker='o', linewidth=2, label='Media semanal')
plt.legend()
plt.title('Downsampling: Diario a Semanal')
plt.tight_layout()

separator_line()
# Diferentes métodos de agregación
agregaciones = {
    'Media': serie_diaria.resample('W').mean(),
    'Suma': serie_diaria.resample('W').sum(),
    'Mínimo': serie_diaria.resample('W').min(),
    'Máximo': serie_diaria.resample('W').max(),
    'Primero': serie_diaria.resample('W').first(),
    'Último': serie_diaria.resample('W').last()
}

# Crear un DataFrame con las diferentes agregaciones
df_agregaciones = pd.DataFrame(agregaciones)
print("\nComparación de métodos de agregación:")
print(df_agregaciones)

separator_line()

# Crear una serie mensual
fechas_mensuales = pd.date_range('2023-01-31', '2023-12-31', freq='M')
valores_mensuales = np.random.normal(0, 1, size=len(fechas_mensuales)).cumsum()
serie_mensual = pd.Series(valores_mensuales, index=fechas_mensuales)

# Upsampling: de mensual a diario
serie_diaria_ffill = serie_mensual.resample('D').ffill()  # Propaga el último valor conocido
serie_diaria_bfill = serie_mensual.resample('D').bfill()  # Usa el próximo valor conocido
serie_diaria_interp = serie_mensual.resample('D').interpolate()  # Interpolación lineal

# Visualizar los diferentes métodos de relleno
fig, ax = plt.subplots(figsize=(12, 6))
serie_mensual.plot(ax=ax, marker='o', markersize=10, linewidth=0, label='Datos mensuales')
serie_diaria_ffill.plot(ax=ax, label='Forward Fill')
serie_diaria_bfill.plot(ax=ax, label='Backward Fill')
serie_diaria_interp.plot(ax=ax, label='Interpolación')
plt.legend()
plt.title('Upsampling: Mensual a Diario')
plt.tight_layout()


separator_line()

# Crear un DataFrame con múltiples métricas
fechas = pd.date_range('2023-01-01', '2023-03-31', freq='D')
df = pd.DataFrame({
    'temperatura': np.random.normal(20, 5, size=len(fechas)),
    'humedad': np.random.normal(60, 10, size=len(fechas)),
    'ventas': np.random.randint(100, 500, size=len(fechas))
}, index=fechas)

# Remuestreo mensual con diferentes agregaciones por columna
df_mensual = df.resample('M').agg({
    'temperatura': 'mean',
    'humedad': ['min', 'max', 'mean'],
    'ventas': 'sum'
})

print("DataFrame remuestreado con agregaciones específicas:")
print(df_mensual)

separator_line()

# Serie con datos de fin de mes
fechas_fin_mes = pd.date_range('2023-01-31', '2023-12-31', freq='M')
valores_fin_mes = np.random.normal(0, 1, size=len(fechas_fin_mes)).cumsum()
serie_fin_mes = pd.Series(valores_fin_mes, index=fechas_fin_mes)

# Cambiar a frecuencia de inicio de mes
serie_inicio_mes = serie_fin_mes.asfreq('MS')
print("\nCambio de frecuencia de fin de mes a inicio de mes:")
print(pd.concat([serie_fin_mes, serie_inicio_mes], axis=1, 
                keys=['Fin de mes', 'Inicio de mes']).head())


separator_line()

# Crear una serie con datos de días hábiles
dias_habiles = pd.date_range('2023-01-01', '2023-01-31', freq='B')
serie_habiles = pd.Series(np.random.randn(len(dias_habiles)).cumsum(), 
                         index=dias_habiles)

# Convertir a serie de todos los días (incluyendo fines de semana)
serie_todos_dias = serie_habiles.asfreq('D', method='ffill')

print("\nSerie original (días hábiles):")
print(serie_habiles.head())
print("\nSerie convertida (todos los días):")
print(serie_todos_dias.head())

separator_line()

# Crear una serie con datos de días hábiles
dias_habiles = pd.date_range('2023-01-01', '2023-01-31', freq='B')
serie_habiles = pd.Series(np.random.randn(len(dias_habiles)).cumsum(), 
                         index=dias_habiles)

# Convertir a serie de todos los días (incluyendo fines de semana)
serie_todos_dias = serie_habiles.asfreq('D', method='ffill')

print("\nSerie original (días hábiles):")
print(serie_habiles.head())
print("\nSerie convertida (todos los días):")
print(serie_todos_dias.head())


separator_line()

# Simular precios diarios de acciones durante un año
fechas_trading = pd.date_range('2023-01-01', '2023-12-31', freq='B')
precios = 100 * (1 + np.random.normal(0.0005, 0.01, size=len(fechas_trading))).cumprod()
serie_precios = pd.Series(precios, index=fechas_trading)

# Calcular rendimientos diarios
rendimientos_diarios = serie_precios.pct_change().dropna()

# Remuestrear a diferentes frecuencias
rendimientos = {
    'Diario': rendimientos_diarios,
    'Semanal': serie_precios.resample('W').last().pct_change().dropna(),
    'Mensual': serie_precios.resample('M').last().pct_change().dropna(),
    'Trimestral': serie_precios.resample('Q').last().pct_change().dropna()
}

# Calcular estadísticas para cada frecuencia
estadisticas = {}
for nombre, serie in rendimientos.items():
    estadisticas[nombre] = {
        'Media (%)': serie.mean() * 100,
        'Desv. Estándar (%)': serie.std() * 100,
        'Mínimo (%)': serie.min() * 100,
        'Máximo (%)': serie.max() * 100
    }

df_estadisticas = pd.DataFrame(estadisticas).T
print("\nEstadísticas de rendimientos por frecuencia:")
print(df_estadisticas)


separator_line()


# Simular datos horarios de temperatura durante un mes
fechas_hora = pd.date_range('2023-07-01', '2023-07-31 23:00:00', freq='H')
# Simular ciclo diario de temperatura (más calor durante el día)
hora_del_dia = fechas_hora.hour
temperatura_base = 20 + 5 * np.sin(np.pi * hora_del_dia / 12)
# Añadir variación aleatoria y tendencia
ruido = np.random.normal(0, 1, size=len(fechas_hora))
tendencia = np.linspace(0, 2, len(fechas_hora))  # Aumento gradual en el mes
temperaturas = temperatura_base + ruido + tendencia

serie_temp = pd.Series(temperaturas, index=fechas_hora)

# Analizar a diferentes escalas temporales
temp_diaria = serie_temp.resample('D').agg(['min', 'mean', 'max'])
temp_semanal = serie_temp.resample('W').agg(['min', 'mean', 'max'])

print("\nEstadísticas diarias de temperatura:")
print(temp_diaria.head())

# Calcular amplitud térmica diaria
temp_diaria['amplitud'] = temp_diaria['max'] - temp_diaria['min']
print("\nAmplitud térmica diaria:")
print(temp_diaria['amplitud'].head())


separator_line()

# Crear serie con un valor atípico
fechas = pd.date_range('2023-01-01', '2023-01-31', freq='H')
valores = np.random.normal(100, 10, size=len(fechas))
# Introducir un valor atípico
valores[350] = 200
serie_con_outlier = pd.Series(valores, index=fechas)

# Calcular media móvil diaria
media_diaria = serie_con_outlier.resample('D').mean()

# Comparar cada valor con la media diaria correspondiente
serie_normalizada = serie_con_outlier.copy()
for fecha, valor in serie_con_outlier.items():
    dia = fecha.floor('D')  # Redondear a día
    media_del_dia = media_diaria.loc[dia]
    serie_normalizada.loc[fecha] = (valor - media_del_dia) / media_diaria.std()

# Detectar valores atípicos (más de 3 desviaciones estándar)
outliers = serie_normalizada[abs(serie_normalizada) > 3]
print("\nValores atípicos detectados:")
print(outliers)

separator_line()

# Crear una serie diaria simple
fechas = pd.date_range('2023-01-01', '2023-01-10', freq='D')
valores = np.arange(1, len(fechas) + 1)
serie = pd.Series(valores, index=fechas)

# Comparar resample() y asfreq() para cambiar a frecuencia de 2 días
comparacion = pd.DataFrame({
    'Original': serie,
    'resample().mean()': serie.resample('2D').mean(),
    'resample().first()': serie.resample('2D').first(),
    'asfreq("2D")': serie.asfreq('2D')
})

print("\nComparación entre resample() y asfreq():")
print(comparacion)


separator_line()