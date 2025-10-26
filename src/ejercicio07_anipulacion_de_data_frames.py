'''
Tienes dos DataFrames con información de ventas y productos. El primer DataFrame ventas contiene las columnas 'id_producto', 'fecha' y 'unidades_vendidas'. El segundo DataFrame productos contiene las columnas 'id_producto', 'nombre' y 'precio'. Debes realizar las siguientes tareas:

Crea los dos DataFrames con los siguientes datos:
ventas: id_producto (A1, A2, A3, A4, A2), fecha (usar fechas consecutivas desde '2023-01-01'), unidades_vendidas (10, 5, 8, 12, 7)
productos: id_producto (A1, A2, A3, A5), nombre ('Laptop', 'Monitor', 'Teclado', 'Mouse'), precio (1200, 300, 100, 50)
Realiza una unión (merge) de tipo 'inner' entre ambos DataFrames usando la columna 'id_producto'.

Realiza una unión (merge) de tipo 'left' entre ambos DataFrames.

Realiza una unión (merge) de tipo 'outer' entre ambos DataFrames.

Crea una nueva columna 'valor_total' en el resultado de la unión 'inner' que multiplique las 'unidades_vendidas' por el 'precio'.

Muestra el resultado de cada operación.
'''
import pandas as pd

# Crear el DataFrame de ventas
data_ventas = {
    'id_producto': ['A1', 'A2', 'A3', 'A4', 'A2'],
    'fecha': pd.date_range(start='2023-01-01', periods=5, freq='D'),
    'unidades_vendidas': [10, 5, 8, 12, 7]
}

df_ventas = pd.DataFrame(data_ventas)

# Crear el DataFrame de productos
data_productos = {
    'id_producto': ['A1', 'A2', 'A3', 'A5'],
    'nombre': ['Laptop', 'Monitor', 'Teclado', 'Mouse'],
    'precio': [1200, 300, 100, 50]
}

df_productos = pd.DataFrame(data_productos)

# Unión (merge) de tipo 'inner'
df_inner = pd.merge(df_ventas, df_productos, on='id_producto', how='inner')
print("Unión (merge) de tipo 'inner':")
print(df_inner)

# Unión (merge) de tipo 'left'
df_left = pd.merge(df_ventas, df_productos, on='id_producto', how='left')
print("\nUnión (merge) de tipo 'left':")
print(df_left)

# Unión (merge) de tipo 'outer'
df_outer = pd.merge(df_ventas, df_productos, on='id_producto', how='outer')
print("\nUnión (merge) de tipo 'outer':")
print(df_outer)

# Crear la columna 'valor_total' en el DataFrame resultante de la unión 'inner'
df_inner['valor_total'] = df_inner['unidades_vendidas'] * df_inner['precio']
print("\nDataFrame 'inner' con la columna 'valor_total':")
print(df_inner)

# --- IGNORE ---
# End of recent edits
