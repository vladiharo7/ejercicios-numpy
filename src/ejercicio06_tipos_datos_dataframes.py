'''
Crea un DataFrame con las siguientes columnas: 'edad' (valores enteros), 'altura' (valores decimales), 'nombre' (texto) y 'activo' (booleanos). Luego realiza las siguientes tareas:

Verifica y muestra los tipos de datos de todas las columnas usando el atributo dtypes
Convierte la columna 'edad' a tipo float64
Convierte la columna 'altura' a tipo int64
Convierte la columna 'nombre' a tipo category
Muestra nuevamente los tipos de datos para verificar los cambios
Calcula y muestra el uso de memoria del DataFrame antes y después de las conversiones usando memory_usage(deep=True)
'''

import pandas as pd

# Crear un diccionario con los datos
data = {
    'edad': [25, 30, 22, 35, 28],
    'altura': [175.5, 180.2, 165.3, 190.0, 170.8],
    'nombre': ['Ana', 'Luis', 'Carlos', 'Marta', 'Sofía'],
    'activo': [True, False, True, True, False]
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Mostrar los tipos de datos iniciales
print("Tipos de datos iniciales:")
print(df.dtypes)

# Calcular y mostrar el uso de memoria antes de las conversiones
memoria_inicial = df.memory_usage(deep=True).sum()
print(f"\nUso de memoria inicial: {memoria_inicial} bytes")

# Convertir la columna 'edad' a float64
df['edad'] = df['edad'].astype('float64')

# Convertir la columna 'altura' a int64
df['altura'] = df['altura'].astype('int64')

# Convertir la columna 'nombre' a category
df['nombre'] = df['nombre'].astype('category')

# Mostrar los tipos de datos después de las conversiones
print("\nTipos de datos después de las conversiones:")
print(df.dtypes)

# Calcular y mostrar el uso de memoria después de las conversiones
memoria_final = df.memory_usage(deep=True).sum()
print(f"\nUso de memoria después de las conversiones: {memoria_final} bytes")

# --- IGNORE ---
# End of recent edits
