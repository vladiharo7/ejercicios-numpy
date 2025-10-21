'''Ejercicio 02: Análisis de Datos Meteorológicos con NumPy'''

import numpy as np

# Simular datos meteorológicos diarios para un mes (temperatura, humedad, presión)
np.random.seed(42)  # Para reproducibilidad
DIAS = 30
temperaturas = np.random.normal(25, 5, DIAS)  # Celsius
humedades = np.random.normal(60, 15, DIAS)    # Porcentaje
presiones = np.random.normal(1013, 5, DIAS)   # hPa

# 1. Usar take() para extraer datos de días específicos (fines de semana)
fines_semana = np.array([5, 6, 12, 13, 19, 20, 26, 27])  # Índices de sábados y domingos
temp_fines_semana = temperaturas.take(fines_semana)
hum_fines_semana = humedades.take(fines_semana)

print(f"Temperatura media en fines de semana: {np.mean(temp_fines_semana):.2f}°C")
print(f"Temperatura media en días laborables: {np.mean(np.delete(temperaturas, fines_semana)):.2f}°C")

# 2. Usar choose() para clasificar los días según su confort
# Definir categorías de confort basadas en temperatura y humedad
indices_confort = np.zeros(DIAS, dtype=int)

# Condiciones de confort (simplificadas):
# 0: Frío (temp < 20)
# 1: Confortable (temp 20-30 y humedad 40-70)
# 2: Caluroso (temp > 30)
# 3: Húmedo (humedad > 70)
indices_confort = np.where(temperaturas < 20, 0, indices_confort)
indices_confort = np.where((temperaturas >= 20) & (temperaturas <= 30) & 
                          (humedades >= 40) & (humedades <= 70), 1, indices_confort)
indices_confort = np.where(temperaturas > 30, 2, indices_confort)
indices_confort = np.where(humedades > 70, 3, indices_confort)

# Mensajes de confort
mensajes = [
    "Día frío, llevar abrigo",
    "Día confortable, perfecto para actividades al aire libre",
    "Día caluroso, mantenerse hidratado",
    "Día húmedo, sensación pegajosa"
]

# Contar días en cada categoría
for i, mensaje in enumerate(mensajes):
    count = np.sum(indices_confort == i)
    print(f"{mensaje}: {count} días")

# 3. Usar compress() para filtrar días con condiciones específicas
# Filtrar días con presión atmosférica baja (posible mal tiempo)
dias_baja_presion = np.compress(presiones < 1010, np.arange(DIAS))
print(f"\nDías con presión atmosférica baja: {dias_baja_presion}")

# Extraer datos completos de esos días
datos_dias_baja_presion = np.column_stack([
    temperaturas.take(dias_baja_presion),
    humedades.take(dias_baja_presion),
    presiones.take(dias_baja_presion)
])

print("\nDatos de días con baja presión (Temp, Humedad, Presión):")
for i, dia in enumerate(dias_baja_presion):
    print(f"Día {dia}: {datos_dias_baja_presion[i][0]:.1f}°C, {datos_dias_baja_presion[i][1]:.1f}%, {datos_dias_baja_presion[i][2]:.1f}hPa")
