'''
Crea un DataFrame de Pandas que contenga información sobre ventas de productos. 
El DataFrame debe tener las siguientes columnas: 'producto', 'precio', 'unidades_vendidas' y 'fecha_venta'.

Incluye al menos 5 productos diferentes con sus respectivos datos. Las fechas de venta deben estar en formato datetime y corresponder al año actual.

Una vez creado el DataFrame, realiza las siguientes operaciones:

Añade una nueva columna llamada 'ingresos_totales' que calcule el producto entre 'precio' y 'unidades_vendidas'.
Muestra los productos ordenados de mayor a menor ingreso total.
Calcula y muestra el precio promedio de todos los productos.
Identifica y muestra el producto con más unidades vendidas.
Puedes comenzar importando las bibliotecas necesarias y creando un diccionario con los datos para construir el DataFrame.
'''

from datetime import datetime
import pandas as pd

# Crear un diccionario con los datos
data = {
    'producto': ['Producto A', 'Producto B', 'Producto C', 'Producto D', 'Producto E'],
    'precio': [10.5, 20.0, 15.75, 30.0, 25.5],
    'unidades_vendidas': [100, 150, 200, 80, 120],
    'fecha_venta': [
        datetime(2024, 1, 15),
        datetime(2024, 2, 20),
        datetime(2024, 3, 10),
        datetime(2024, 4, 5),
        datetime(2024, 5, 25)
    ]
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Añadir la columna 'ingresos_totales'
df['ingresos_totales'] = df['precio'] * df['unidades_vendidas']
print("DataFrame con ingresos totales:")
print(df)

# Mostrar los productos ordenados de mayor a menor ingreso total
df_sorted = df.sort_values(by='ingresos_totales', ascending=False)
print("\nProductos ordenados por ingresos totales:")
print(df_sorted)

# Calcular y mostrar el precio promedio de todos los productos
precio_promedio = df['precio'].mean()
print(f"\nPrecio promedio de todos los productos: {precio_promedio}")

# Identificar y mostrar el producto con más unidades vendidas
producto_mas_vendido = df.loc[df['unidades_vendidas'].idxmax()]
print("\nProducto con más unidades vendidas:")
print(producto_mas_vendido)

# --- IGNORE ---
# End of recent edits
