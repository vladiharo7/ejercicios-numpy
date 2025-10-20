'''
Ejemplo de manipulación de datos con Pandas
'''
import pandas as pd

# Ejemplo básico de manipulación de datos con Pandas
data = {
    'Nombre': ['Ana', 'Carlos', 'Elena', 'David'],
    'Edad': [28, 32, 25, 41],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla'],
    'Puntuación': [85, 92, 78, 96]
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Filtrar datos
mayores_30 = df[df['Edad'] > 30]

# Agrupar y calcular estadísticas
promedio_por_ciudad = df.groupby('Ciudad')['Puntuación'].mean()

print("Personas mayores de 30 años:")
print(mayores_30)
print("\nPuntuación promedio por ciudad:")
print(promedio_por_ciudad)

