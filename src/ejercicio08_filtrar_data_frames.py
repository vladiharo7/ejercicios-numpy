'''
Crea un DataFrame de ventas con las siguientes columnas: 'producto', 'categoria', 'precio' y 'stock'. Incluye al menos 8 productos de diferentes categorías (por ejemplo: 'Electrónica', 'Ropa', 'Hogar'). Luego, realiza las siguientes operaciones de filtrado:

Filtra los productos que pertenecen a la categoría 'Electrónica'.
Encuentra los productos con precio mayor a 50 y stock menor a 20.
Selecciona los productos que contengan la letra 'a' en su nombre.
Utiliza el método query() para encontrar productos de la categoría 'Hogar' con precio menor a 30.
Muestra el resultado de cada filtrado por separado.
'''

import pandas as pd

# Crear el DataFrame de ventas
data = {
    'producto': ['Televisor', 'Camiseta', 'Sofá', 'Laptop', 'Pantalones', 'Microondas', 'Zapatos', 'Lámpara'],
    'categoria': ['Electrónica', 'Ropa', 'Hogar', 'Electrónica', 'Ropa', 'Hogar', 'Ropa', 'Hogar'],
    'precio': [300, 20, 150, 800, 40, 70, 60, 25],
    'stock': [15, 50, 10, 5, 30, 8, 25, 12]
}

ventas_df = pd.DataFrame(data)
print("DataFrame de ventas:")
print(ventas_df)

# 1. Filtrar productos de la categoría 'Electrónica'
electronica_df = ventas_df[ventas_df['categoria'] == 'Electrónica']
electronica_df2 = ventas_df.query("categoria == 'Electrónica'")
print("\nProductos de la categoría 'Electrónica':")
print(electronica_df)
print("\nProductos de la categoría 'Electrónica 2':")
print(electronica_df2)

# 2. Encontrar productos con precio mayor a 50 y stock menor a 20
precio_stock_df = ventas_df[(ventas_df['precio'] > 50) & (ventas_df['stock'] < 20)]
print("\nProductos con precio mayor a 50 y stock menor a 20:")
print(precio_stock_df)

# 3. Seleccionar productos que contengan la letra 'a' en su nombre
productos_con_a_df = ventas_df[ventas_df['producto'].str.contains('a', case=False)]
print("\nProductos que contienen la letra 'a' en su nombre:")
print(productos_con_a_df)

# 4. Utilizar query() para encontrar productos de la categoría 'Hogar' con precio menor a 30
hogar_barato_df = ventas_df.query("categoria == 'Hogar' and precio < 30")
print("\nProductos de la categoría 'Hogar' con precio menor a 30:")
print(hogar_barato_df)

# Fin del código
