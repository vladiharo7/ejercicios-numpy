import pandas as pd
import numpy as np

def separator_line():
    print("\n" + "-" * 50 + "\n")


# Creando una Serie con datos de texto
nombres = pd.Series(['Ana García', 'Juan Pérez', 'María Rodríguez', 'Carlos López'])

# Accediendo al accesorio .str
nombres_mayusculas = nombres.str.upper()
print(nombres_mayusculas)

separator_line()

# Convertir a mayúsculas
nombres.str.upper()

# Convertir a minúsculas
nombres.str.lower()

# Capitalizar (primera letra en mayúscula)
nombres.str.capitalize()

# Convertir a formato título (primera letra de cada palabra en mayúscula)
nombres.str.title()

separator_line()

# Extraer los primeros 4 caracteres
nombres.str[:4]

# Eliminar espacios en blanco al inicio y final
nombres.str.strip()

# Eliminar espacios a la izquierda o derecha
nombres.str.lstrip()  # izquierda
nombres.str.rstrip()  # derecha

separator_line()

# Verificar si las cadenas contienen un patrón específico
contiene_ana = nombres.str.contains('Ana')
print(contiene_ana)

# Contar ocurrencias de un patrón
ocurrencias_a = nombres.str.count('a')
print(ocurrencias_a)

# Verificar si las cadenas comienzan con un patrón
comienza_con_m = nombres.str.startswith('M')
print(comienza_con_m)

# Verificar si las cadenas terminan con un patrón
termina_con_ez = nombres.str.endswith('ez')
print(termina_con_ez)

separator_line()

# Dividir cadenas en listas basadas en un separador
nombres_divididos = nombres.str.split(' ')
print(nombres_divididos)

# Acceder a elementos específicos después de dividir
apellidos = nombres.str.split(' ').str[1]
print(apellidos)

# Reemplazar patrones
nombres_reemplazados = nombres.str.replace('García', 'Gómez')
print(nombres_reemplazados)

separator_line()

# Creando un DataFrame con datos de productos
datos = {
    'producto': ['Laptop HP 15"', 'Monitor Samsung 24"', 'Teclado mecánico RGB', 'Mouse inalámbrico'],
    'categoria': ['Portátiles', 'Monitores', 'Periféricos', 'Periféricos'],
    'precio': [799.99, 249.50, 89.99, 35.50]
}

df = pd.DataFrame(datos)
print(df)

separator_line()

# Extraer la marca del producto
df['marca'] = df['producto'].str.split(' ').str[0]

# Extraer el tamaño de los productos que lo tienen (con comillas)
df['tamaño'] = df['producto'].str.extract(r'(\d+")') 

# Convertir categorías a minúsculas
df['categoria'] = df['categoria'].str.lower()

# Verificar qué productos son periféricos
df['es_periferico'] = df['categoria'].str.contains('periféricos')

print(df)

separator_line()
# Serie con valores nulos
textos = pd.Series(['python', 'pandas', None, 'numpy', pd.NA])

# Las operaciones .str ignoran los valores nulos
mayusculas = textos.str.upper()
print(mayusculas)
separator_line()

# Ejemplo de encadenamiento de operaciones
resultado = df['producto'].str.lower().str.replace('"', '').str.split(' ').str[0:2].str.join('_')
print(resultado)

separator_line()

# Comparación de rendimiento
import time

# Método tradicional con bucle
start = time.time()
resultado_bucle = []
for nombre in nombres:
    resultado_bucle.append(nombre.upper())
tiempo_bucle = time.time() - start

# Método vectorizado con .str
start = time.time()
resultado_str = nombres.str.upper()
tiempo_str = time.time() - start

print(f"Tiempo con bucle: {tiempo_bucle:.6f} segundos")
print(f"Tiempo con .str: {tiempo_str:.6f} segundos")
separator_line()

# Filtrar productos que contienen la palabra "inalámbrico"
productos_inalambricos = df[df['producto'].str.contains('inalámbrico')]
print(productos_inalambricos)

# Agrupar por la primera letra de cada categoría
agrupado = df.groupby(df['categoria'].str[0]).count()
print(agrupado)
separator_line()

# Aplicar operaciones de texto a múltiples columnas
columnas_texto = ['producto', 'categoria']
for col in columnas_texto:
    df[f"{col}_longitud"] = df[col].str.len()
    
print(df)

separator_line()

# Creación directa de una Serie categórica
categorias = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'], dtype='category')
print(categorias)

# Convertir una columna existente a tipo categórico
df = pd.DataFrame({
    'ciudad': ['Madrid', 'Barcelona', 'Sevilla', 'Madrid', 'Valencia', 'Barcelona'],
    'temperatura': [32, 29, 35, 30, 31, 28]
})

df['ciudad'] = df['ciudad'].astype('category')
print(df.dtypes)

separator_line()

# Comparación de memoria utilizada
ciudades_objeto = df['ciudad'].copy().astype('object')
ciudades_categoria = df['ciudad'].copy().astype('category')

print(f"Memoria como object: {ciudades_objeto.memory_usage(deep=True)} bytes")
print(f"Memoria como category: {ciudades_categoria.memory_usage(deep=True)} bytes")

separator_line()

# Acceder a las categorías
print("Categorías disponibles:", df['ciudad'].cat.categories)

# Obtener códigos internos
print("Códigos internos:", df['ciudad'].cat.codes)

# Verificar si las categorías están ordenadas
print("¿Categorías ordenadas?:", df['ciudad'].cat.ordered)
separator_line()

# Añadir nuevas categorías
df['ciudad'] = df['ciudad'].cat.add_categories(['Bilbao', 'Málaga'])
print(df['ciudad'].cat.categories)

# Eliminar categorías no utilizadas
df['ciudad'] = df['ciudad'].cat.remove_unused_categories()
print(df['ciudad'].cat.categories)

# Renombrar categorías
df['ciudad'] = df['ciudad'].cat.rename_categories({'Madrid': 'MAD', 'Barcelona': 'BCN', 
                                                  'Sevilla': 'SVQ', 'Valencia': 'VLC'})
print(df['ciudad'])

separator_line()

# Crear categorías ordenadas
tallas = pd.Series(['M', 'XL', 'S', 'L', 'M', 'S'], 
                  dtype=pd.CategoricalDtype(categories=['XS', 'S', 'M', 'L', 'XL'], ordered=True))

print(tallas)
print("¿Es M mayor que S?:", tallas[0] > tallas[2])  # Compara 'M' > 'S'

# Convertir una columna existente a categórica ordenada
niveles_educativos = ['Primaria', 'Secundaria', 'Universidad', 'Máster', 'Doctorado']
df = pd.DataFrame({
    'educacion': ['Universidad', 'Secundaria', 'Máster', 'Primaria', 'Doctorado']
})

df['educacion'] = pd.Categorical(df['educacion'], 
                                categories=niveles_educativos, 
                                ordered=True)

# Ahora podemos ordenar por nivel educativo
df_ordenado = df.sort_values('educacion')
print(df_ordenado)

separator_line()

# Filtrar por nivel educativo
nivel_alto = df[df['educacion'] > 'Secundaria']
print("Niveles superiores a Secundaria:")
print(nivel_alto)

# Encontrar el nivel mínimo y máximo
print("Nivel mínimo:", df['educacion'].min())
print("Nivel máximo:", df['educacion'].max())

separator_line()
# One-hot encoding (variables dummy)
dummies = pd.get_dummies(df['educacion'], prefix='edu')
df_con_dummies = pd.concat([df, dummies], axis=1)
print(df_con_dummies)

# Codificación ordinal (usando los códigos internos)
df['educacion_codigo'] = df['educacion'].cat.codes
print(df[['educacion', 'educacion_codigo']])

separator_line()

# Crear un DataFrame con datos categóricos
datos = pd.DataFrame({
    'departamento': pd.Categorical(['Ventas', 'IT', 'Marketing', 'Ventas', 'IT', 'Marketing', 'Ventas']),
    'rendimiento': pd.Categorical(['Alto', 'Medio', 'Bajo', 'Medio', 'Alto', 'Alto', 'Bajo'],
                                 categories=['Bajo', 'Medio', 'Alto'], ordered=True),
    'salario': [35000, 48000, 40000, 37000, 52000, 45000, 33000]
})

# Tabla de contingencia
tabla = pd.crosstab(datos['departamento'], datos['rendimiento'])
print("Tabla de contingencia:")
print(tabla)

# Frecuencias relativas
tabla_porcentaje = pd.crosstab(datos['departamento'], datos['rendimiento'], normalize='index')
print("\nPorcentajes por departamento:")
print(tabla_porcentaje)

# Estadísticas descriptivas agrupadas por categorías
estadisticas = datos.groupby('departamento')['salario'].describe()
print("\nEstadísticas por departamento:")
print(estadisticas)

separator_line()

# De categórico a string
# ciudades_str = df['ciudad'].astype(str)
# print(type(ciudades_str[0]))

# De categórico a objeto
# ciudades_obj = df['ciudad'].astype('object')
# print(ciudades_obj.dtype)

# De string a categórico con parámetros específicos
colores = pd.Series(['rojo', 'verde', 'azul', 'rojo', 'verde'])
colores_cat = colores.astype(pd.CategoricalDtype(categories=['rojo', 'verde', 'azul', 'amarillo'], ordered=False))
print(colores_cat)

separator_line()

# Crear un DataFrame con columnas potencialmente categóricas
datos_ventas = pd.DataFrame({
    'producto': ['Laptop', 'Smartphone', 'Tablet', 'Laptop', 'Smartphone'] * 1000,
    'tienda': ['Central', 'Norte', 'Sur', 'Central', 'Este'] * 1000,
    'unidades': np.random.randint(1, 50, 5000)
})

# Convertir automáticamente columnas con pocos valores únicos a categóricas
datos_optimizados = datos_ventas.convert_dtypes()
print(datos_optimizados.dtypes)

# Otra opción es usar select_dtypes para identificar columnas object
columnas_objeto = datos_ventas.select_dtypes(include=['object']).columns
for col in columnas_objeto:
    # Convertir a categórico si hay menos de 50 valores únicos
    if datos_ventas[col].nunique() < 50:
        datos_ventas[col] = datos_ventas[col].astype('category')

print(datos_ventas.dtypes)

separator_line()

# Análisis de encuestas
encuesta = pd.DataFrame({
    'edad': [25, 34, 42, 51, 29, 37, 45],
    'genero': pd.Categorical(['M', 'F', 'F', 'M', 'F', 'M', 'F']),
    'satisfaccion': pd.Categorical(['Alta', 'Media', 'Baja', 'Media', 'Alta', 'Baja', 'Media'],
                                  categories=['Baja', 'Media', 'Alta'], ordered=True)
})

# Análisis por género
por_genero = pd.crosstab(encuesta['genero'], encuesta['satisfaccion'], normalize='index')
print("Satisfacción por género (%):")
print(por_genero * 100)

# Filtrado combinado
jovenes_satisfechos = encuesta[(encuesta['edad'] < 40) & (encuesta['satisfaccion'] > 'Baja')]
print("\nPersonas jóvenes con satisfacción media o alta:")
print(jovenes_satisfechos)

separator_line()

# Creamos un DataFrame de ejemplo con datos textuales
datos = pd.DataFrame({
    'texto': [
        'El precio es $1,500.00',
        'Contacto: info@empresa.com',
        'Teléfono: 91-234-5678',
        'Código: ABC-123-XYZ',
        'Fecha: 2023-05-15'
    ]
})

separator_line()

# Buscar filas que contienen direcciones de correo electrónico
tiene_email = datos['texto'].str.contains(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
print("Filas con emails:")
print(datos[tiene_email])

# Buscar filas que contienen precios en dólares
tiene_precio = datos['texto'].str.contains(r'\$\d+(?:,\d+)*(?:\.\d+)?')
print("\nFilas con precios:")
print(datos[tiene_precio])

separator_line()

# Búsqueda sin distinguir mayúsculas/minúsculas
contiene_codigo = datos['texto'].str.contains('código', case=False)
print("Contiene 'código' (ignorando mayúsculas):")
print(datos[contiene_codigo])

# Usar regex=False para buscar texto literal (no como expresión regular)
contiene_parentesis = datos['texto'].str.contains('(', regex=False)
separator_line()

# Identificar filas que comienzan con "El" o "La"
comienza_articulo = datos['texto'].str.match(r'(El|La)\s')
print("Comienza con artículo:")
print(datos[comienza_articulo])

separator_line()

# Extraer todos los números de cada texto
numeros = datos['texto'].str.findall(r'\d+')
print("Números encontrados:")
print(numeros)

# Extraer palabras que comienzan con mayúscula
palabras_mayuscula = datos['texto'].str.findall(r'\b[A-Z][a-z]+\b')
print("\nPalabras con mayúscula inicial:")
print(palabras_mayuscula)

separator_line()

# Extraer el primer grupo que coincida con el patrón
codigos = datos['texto'].str.extract(r'Código: ([A-Z]+-\d+-[A-Z]+)')
print("Códigos extraídos:")
print(codigos)

# Extraer múltiples grupos
# Formato: (parte1)-(parte2)-(parte3)
partes_codigo = datos['texto'].str.extract(r'Código: ([A-Z]+)-(\d+)-([A-Z]+)')
partes_codigo.columns = ['Prefijo', 'Número', 'Sufijo']
print("\nPartes del código:")
print(partes_codigo)
separator_line()

# Extraer todos los pares de letras y números
pares = datos['texto'].str.extractall(r'([A-Z])(\d)')
print("Pares de letra-número:")
print(pares)
separator_line()

# Reemplazar formatos de precio
texto_modificado = datos['texto'].str.replace(r'\$(\d+),(\d+)\.(\d+)', r'\1\2.\3 USD', regex=True)
print("Precios reformateados:")
print(texto_modificado)

# Ocultar información sensible (emails)
anonimizado = datos['texto'].str.replace(r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 
                                        r'***@\2', regex=True)
print("\nEmails anonimizados:")
print(anonimizado)

separator_line()
# DataFrame con datos inconsistentes
productos = pd.DataFrame({
    'descripcion': [
        'Laptop HP 15.6" - 8GB RAM',
        'Monitor Samsung 24 pulgadas',
        'Teclado inalámbrico (USB)',
        'Mouse gaming - 6000 DPI',
        'SSD 500GB Samsung'
    ],
    'precio': [
        '$899.99',
        '249,50 €',
        '35.99$',
        '€ 59,95',
        '$120'
    ]
})

# Estandarizar formato de precios a valor numérico
def extraer_precio(texto):
    # Eliminar símbolos de moneda y espacios
    limpio = pd.Series(texto).str.replace(r'[$€\s]', '', regex=True)
    # Convertir comas a puntos para decimales
    return limpio.str.replace(',', '.', regex=False)

productos['precio_num'] = extraer_precio(productos['precio']).astype(float)
print("Precios estandarizados:")
print(productos[['precio', 'precio_num']])

# Extraer tamaños de pantalla
productos['tamaño_pantalla'] = productos['descripcion'].str.extract(r'(\d+(?:\.\d+)?)\s*(?:"|pulgadas)')
print("\nTamaños de pantalla:")
print(productos[['descripcion', 'tamaño_pantalla']])
separator_line()

# Logs de servidor web
logs = pd.Series([
    '192.168.1.1 - - [15/May/2023:10:12:01 +0200] "GET /index.html HTTP/1.1" 200 4523',
    '10.0.0.5 - - [15/May/2023:10:12:05 +0200] "POST /login HTTP/1.1" 302 0',
    '192.168.1.1 - - [15/May/2023:10:12:10 +0200] "GET /dashboard HTTP/1.1" 200 18432',
    '10.0.0.8 - - [15/May/2023:10:12:15 +0200] "GET /images/logo.png HTTP/1.1" 404 0'
])

# Extraer componentes del log
log_parts = logs.str.extract(
    r'(\d+\.\d+\.\d+\.\d+).*\[(.*?)\].*"(\w+)\s+(.*?)\s+HTTP.*"\s+(\d+)\s+(\d+)'
)
log_parts.columns = ['IP', 'Fecha', 'Método', 'Ruta', 'Código', 'Tamaño']
print("Análisis de logs:")
print(log_parts)

# Filtrar por código de respuesta
errores_404 = log_parts[log_parts['Código'] == '404']
print("\nSolicitudes con error 404:")
print(errores_404)
separator_line()

# DataFrame con datos de contacto
contactos = pd.DataFrame({
    'nombre': ['Ana García', 'Juan Pérez', 'María Rodríguez', 'Carlos López', 'Laura Martínez'],
    'email': ['ana@ejemplo.com', 'juanperez@mail.org', 'maria_rodriguez', 'carlos@empresa.es', 'laura@.com'],
    'telefono': ['612345678', '+34 612 345 678', '6123-4567', '(612)345678', '612 34 56 78']
})

# Validar formato de email
patron_email = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
contactos['email_valido'] = contactos['email'].str.match(patron_email)

# Validar formato de teléfono español (9 dígitos, puede tener prefijo +34)
patron_telefono = r'^(?:\+34\s?)?(?:\(?\d{3}\)?[\s-]?)?(?:\d{2,3}[\s-]?){2,3}\d{2}$'
contactos['telefono_valido'] = contactos['telefono'].str.match(patron_telefono)

print("Validación de datos de contacto:")
print(contactos)

separator_line()

# Agrupar por tipo de solicitud HTTP (del ejemplo de logs)
metodos_por_codigo = log_parts.groupby(['Método', 'Código']).size().unstack(fill_value=0)
print("Conteo de métodos HTTP por código de respuesta:")
print(metodos_por_codigo)

# Aplicar regex a múltiples columnas
def extraer_dominio(email):
    if pd.isna(email):
        return np.nan
    match = pd.Series([email]).str.extract(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$')
    return match.iloc[0, 0] if not match.empty and not pd.isna(match.iloc[0, 0]) else np.nan

# Aplicar a una columna
contactos['dominio'] = contactos['email'].apply(extraer_dominio)
print("\nDominios de email extraídos:")
print(contactos[['email', 'dominio']])

separator_line()

# Ejemplo de optimización para datasets grandes
# Primero filtramos con un método más rápido
datos_grandes = pd.DataFrame({'texto': ['texto largo...'] * 1000000})

# Enfoque ineficiente
# emails = datos_grandes['texto'].str.extract(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Enfoque más eficiente
posibles_emails = datos_grandes['texto'].str.contains('@', regex=False)
# Solo aplicamos regex a las filas que contienen '@'
emails = datos_grandes.loc[posibles_emails, 'texto'].str.extract(
    r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
)

separator_line()

# Ejemplo de datos semi-estructurados: información de productos
descripciones = pd.Series([
    "Producto: Laptop Lenovo ThinkPad X1 | Precio: 1299.99€ | Stock: 15 unidades",
    "Producto: Monitor Dell UltraSharp 27\" | Precio: 449.50€ | Stock: 8 unidades",
    "Producto: Teclado Logitech MX Keys | Precio: 109.95€ | Stock: 23 unidades",
    "Producto: Ratón Logitech MX Master 3 | Precio: 89.99€ | Stock: 12 unidades"
])

separator_line()

# Extraer nombre, precio y stock de cada producto
info_productos = descripciones.str.extract(
    r'Producto: (.*?) \| Precio: (\d+\.\d+)€ \| Stock: (\d+) unidades'
)
info_productos.columns = ['Nombre', 'Precio', 'Stock']

# Convertir tipos de datos
info_productos['Precio'] = info_productos['Precio'].astype(float)
info_productos['Stock'] = info_productos['Stock'].astype(int)

print(info_productos)
separator_line()

# Datos en formato clave-valor
metadatos = pd.Series([
    "autor=John Smith;fecha=2023-05-10;categoría=Tecnología;visitas=1250",
    "autor=Ana García;fecha=2023-05-12;categoría=Ciencia;visitas=980",
    "autor=Carlos López;fecha=2023-05-15;categoría=Tecnología;visitas=1540",
    "autor=María Rodríguez;fecha=2023-05-18;categoría=Salud;visitas=2100"
])

# Función para convertir texto clave-valor a diccionario
def parse_metadata(text):
    pairs = text.split(';')
    result = {}
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            result[key] = value
    return result

# Aplicar la función y expandir el resultado en columnas
df_metadatos = metadatos.apply(parse_metadata).apply(pd.Series)
print(df_metadatos)

# Convertir tipos de datos apropiados
df_metadatos['visitas'] = df_metadatos['visitas'].astype(int)
df_metadatos['fecha'] = pd.to_datetime(df_metadatos['fecha'])

separator_line()

import json

# Texto con datos JSON incrustados
registros_json = pd.Series([
    "ID: 1001 | Datos: {\"nombre\": \"Juan\", \"edad\": 28, \"intereses\": [\"programación\", \"música\"]}",
    "ID: 1002 | Datos: {\"nombre\": \"Ana\", \"edad\": 34, \"intereses\": [\"viajes\", \"fotografía\", \"cocina\"]}",
    "ID: 1003 | Datos: {\"nombre\": \"Carlos\", \"edad\": 42, \"intereses\": [\"deportes\", \"cine\"]}"
])

# Extraer el ID y la parte JSON
partes = registros_json.str.extract(r'ID: (\d+) \| Datos: (.*)')
partes.columns = ['ID', 'JSON']

# Parsear el JSON a diccionarios
def parse_json(json_str):
    try:
        return json.loads(json_str)
    except:
        return {}

# Crear DataFrame con los datos JSON normalizados
df_usuarios = pd.json_normalize(partes['JSON'].apply(parse_json))

# Combinar con los IDs
df_usuarios['ID'] = partes['ID']

print(df_usuarios)
separator_line()

# Texto con formato tabular
tabla_texto = """
Región | Ventas Q1 | Ventas Q2 | Ventas Q3 | Ventas Q4
Norte  | 12500     | 14200     | 15100     | 16800
Sur    | 9800      | 10500     | 11200     | 12100
Este   | 11300     | 12700     | 13500     | 14900
Oeste  | 10200     | 11800     | 12600     | 13700
"""

# Convertir texto a DataFrame
lineas = [line.strip() for line in tabla_texto.strip().split('\n')]
encabezados = [col.strip() for col in lineas[0].split('|')]
datos = []

for linea in lineas[1:]:
    valores = [val.strip() for val in linea.split('|')]
    datos.append(valores)

df_ventas = pd.DataFrame(datos, columns=encabezados)

# Convertir columnas numéricas
columnas_ventas = [col for col in df_ventas.columns if 'Ventas' in col]
for col in columnas_ventas:
    df_ventas[col] = pd.to_numeric(df_ventas[col])

print(df_ventas)
separator_line()

# Simulación de texto HTML extraído
html_texto = pd.Series([
    "<div class='producto'><h2>Smartphone Galaxy S21</h2><p class='precio'>799.99€</p><span class='stock'>Disponible</span></div>",
    "<div class='producto'><h2>iPhone 13</h2><p class='precio'>899.99€</p><span class='stock'>Agotado</span></div>",
    "<div class='producto'><h2>Xiaomi Mi 11</h2><p class='precio'>699.99€</p><span class='stock'>Disponible</span></div>"
])

# Extraer nombre del producto
nombres = html_texto.str.extract(r'<h2>(.*?)</h2>')

# Extraer precio
precios = html_texto.str.extract(r'<p class=\'precio\'>(.*?)</p>')
precios = precios[0].str.replace('€', '').astype(float)

# Extraer disponibilidad
disponibilidad = html_texto.str.extract(r'<span class=\'stock\'>(.*?)</span>')

# Crear DataFrame con la información extraída
df_productos = pd.DataFrame({
    'Producto': nombres[0],
    'Precio': precios,
    'Disponibilidad': disponibilidad[0]
})

print(df_productos)

separator_line()

# Continuando con el ejemplo de ventas por región y trimestre
# Transformar de formato ancho a largo
df_ventas_largo = df_ventas.melt(
    id_vars=['Región'],
    value_vars=[col for col in df_ventas.columns if 'Ventas' in col],
    var_name='Trimestre',
    value_name='Ventas'
)

# Extraer solo el número del trimestre
df_ventas_largo['Trimestre'] = df_ventas_largo['Trimestre'].str.extract(r'Q(\d)')

print(df_ventas_largo)

# Transformar de vuelta a formato ancho
df_ventas_ancho = df_ventas_largo.pivot(
    index='Región',
    columns='Trimestre',
    values='Ventas'
)
df_ventas_ancho.columns = [f'Q{col}' for col in df_ventas_ancho.columns]
df_ventas_ancho = df_ventas_ancho.reset_index()

print(df_ventas_ancho)
separator_line()

# Ejemplo de textos con entidades nombradas
noticias = pd.Series([
    "Apple lanza nuevo iPhone en San Francisco durante evento especial",
    "Microsoft anuncia colaboración con OpenAI para mejorar Bing",
    "El presidente Pedro Sánchez se reúne con líderes europeos en Bruselas",
    "Google presenta Pixel 7 con nuevas funciones de inteligencia artificial"
])

# Extraer empresas tecnológicas
empresas = noticias.str.extractall(r'(Apple|Microsoft|Google|OpenAI)')
print("Empresas mencionadas:")
print(empresas)

# Extraer ciudades
ciudades = noticias.str.extractall(r'(San Francisco|Bruselas)')
print("\nCiudades mencionadas:")
print(ciudades)

# Crear un DataFrame con conteo de menciones
def extraer_entidades(texto):
    entidades = {
        'empresas': [],
        'ciudades': [],
        'personas': []
    }
    
    # Patrones para cada tipo de entidad
    patrones = {
        'empresas': r'(Apple|Microsoft|Google|OpenAI)',
        'ciudades': r'(San Francisco|Bruselas)',
        'personas': r'(Pedro Sánchez)'
    }
    
    for tipo, patron in patrones.items():
        matches = pd.Series([texto]).str.extractall(patron)
        if not matches.empty:
            entidades[tipo] = matches[0].tolist()
    
    return entidades

# Aplicar la función a cada noticia
entidades_df = pd.DataFrame(noticias.apply(extraer_entidades).tolist())
print("\nEntidades extraídas:")
print(entidades_df)

separator_line()

# Ejemplo de logs de aplicación
logs = pd.Series([
    "2023-05-20 08:15:23 INFO [UserService] Usuario 'jsmith' ha iniciado sesión desde 192.168.1.45",
    "2023-05-20 08:16:45 WARNING [SecurityModule] Intento fallido de autenticación para usuario 'admin' desde 203.0.113.42",
    "2023-05-20 08:17:12 ERROR [DatabaseService] Conexión perdida con la base de datos 'usuarios_prod'",
    "2023-05-20 08:18:30 INFO [FileService] Usuario 'jsmith' ha subido archivo 'informe_mayo.pdf' (2.4 MB)"
])

# Extraer componentes del log
componentes_log = logs.str.extract(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) \[(\w+)\] (.*)'
)
componentes_log.columns = ['Timestamp', 'Nivel', 'Servicio', 'Mensaje']

# Convertir timestamp a datetime
componentes_log['Timestamp'] = pd.to_datetime(componentes_log['Timestamp'])

print(componentes_log)

# Extraer información adicional de los mensajes
# Extraer nombres de usuario
componentes_log['Usuario'] = componentes_log['Mensaje'].str.extract(r"Usuario '(\w+)'")

# Extraer direcciones IP
componentes_log['IP'] = componentes_log['Mensaje'].str.extract(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')

# Extraer nombres de archivos
componentes_log['Archivo'] = componentes_log['Mensaje'].str.extract(r"archivo '([^']+)'")

print("\nInformación extraída de logs:")
print(componentes_log[['Timestamp', 'Nivel', 'Usuario', 'IP', 'Archivo']])

separator_line()

# Coordenadas en diferentes formatos
ubicaciones = pd.Series([
    "40°26'46\"N 3°42'10\"W",
    "48.8584° N, 2.2945° E",
    "51°30'26\"N 0°7'39\"W",
    "40.7128° N, 74.0060° W"
])

# Función para convertir coordenadas a formato decimal
def normalizar_coordenadas(coord_str):
    # Patrón para formato DMS (grados, minutos, segundos)
    dms_pattern = r'(\d+)°(\d+)\'(\d+)\"([NS])\s+(\d+)°(\d+)\'(\d+)\"([EW])'
    # Patrón para formato decimal
    dec_pattern = r'(\d+\.\d+)°\s*([NS]),\s*(\d+\.\d+)°\s*([EW])'
    
    # Intentar formato DMS
    dms_match = pd.Series([coord_str]).str.extract(dms_pattern)
    if not dms_match.iloc[0, 0] is None:
        lat_deg, lat_min, lat_sec, lat_dir = dms_match.iloc[0, 0:4]
        lon_deg, lon_min, lon_sec, lon_dir = dms_match.iloc[0, 4:8]
        
        lat = float(lat_deg) + float(lat_min)/60 + float(lat_sec)/3600
        if lat_dir == 'S':
            lat = -lat
            
        lon = float(lon_deg) + float(lon_min)/60 + float(lon_sec)/3600
        if lon_dir == 'W':
            lon = -lon
            
        return pd.Series({'latitud': lat, 'longitud': lon})
    
    # Intentar formato decimal
    dec_match = pd.Series([coord_str]).str.extract(dec_pattern)
    if not dec_match.iloc[0, 0] is None:
        lat, lat_dir, lon, lon_dir = dec_match.iloc[0]
        
        lat = float(lat)
        if lat_dir == 'S':
            lat = -lat
            
        lon = float(lon)
        if lon_dir == 'W':
            lon = -lon
            
        return pd.Series({'latitud': lat, 'longitud': lon})
    
    return pd.Series({'latitud': np.nan, 'longitud': np.nan})

# Aplicar la función a todas las ubicaciones
coordenadas_normalizadas = ubicaciones.apply(normalizar_coordenadas)
print("Coordenadas normalizadas:")
print(coordenadas_normalizadas)

separator_line()

# Fechas en diferentes formatos
fechas_texto = pd.Series([
    "Publicado el 15 de mayo de 2023",
    "Fecha: 2023/05/16",
    "Actualizado: 17-05-2023",
    "Creado: mayo 18, 2023"
])

# Extraer fechas con diferentes patrones
def extraer_fecha(texto):
    # Patrón para "15 de mayo de 2023"
    patron1 = r'(\d{1,2}) de (\w+) de (\d{4})'
    # Patrón para "2023/05/16"
    patron2 = r'(\d{4})/(\d{2})/(\d{2})'
    # Patrón para "17-05-2023"
    patron3 = r'(\d{1,2})-(\d{1,2})-(\d{4})'
    # Patrón para "mayo 18, 2023"
    patron4 = r'(\w+) (\d{1,2}), (\d{4})'
    
    # Diccionario para convertir nombres de meses a números
    meses = {
        'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
        'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
        'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
    }
    
    # Intentar cada patrón
    match1 = pd.Series([texto]).str.extract(patron1)
    if not match1.iloc[0, 0] is None:
        dia, mes, año = match1.iloc[0]
        mes_num = meses.get(mes.lower(), '01')
        return f"{año}-{mes_num}-{int(dia):02d}"
    
    match2 = pd.Series([texto]).str.extract(patron2)
    if not match2.iloc[0, 0] is None:
        año, mes, dia = match2.iloc[0]
        return f"{año}-{mes}-{dia}"
    
    match3 = pd.Series([texto]).str.extract(patron3)
    if not match3.iloc[0, 0] is None:
        dia, mes, año = match3.iloc[0]
        return f"{año}-{int(mes):02d}-{int(dia):02d}"
    
    match4 = pd.Series([texto]).str.extract(patron4)
    if not match4.iloc[0, 0] is None:
        mes, dia, año = match4.iloc[0]
        mes_num = meses.get(mes.lower(), '01')
        return f"{año}-{mes_num}-{int(dia):02d}"
    
    return None

# Aplicar la función y convertir a datetime
# fechas_extraidas = fechas_texto.apply(extraer_fecha)
# fechas_datetime = pd.to_datetime(fechas_extraidas)
print("Fechas normalizadas:")
# print(fechas_datetime)
separator_line()

# Texto con estructura jerárquica
jerarquia_texto = pd.Series([
    "Departamento: Ventas > Equipo: Europa > Miembro: Ana García",
    "Departamento: Ventas > Equipo: América > Miembro: Juan Pérez",
    "Departamento: IT > Equipo: Desarrollo > Miembro: Carlos López",
    "Departamento: IT > Equipo: Infraestructura > Miembro: María Rodríguez"
])

# Extraer los niveles jerárquicos
jerarquia_df = jerarquia_texto.str.extract(
    r'Departamento: (.*?) > Equipo: (.*?) > Miembro: (.*)'
)
jerarquia_df.columns = ['Departamento', 'Equipo', 'Miembro']
print(jerarquia_df)

# Crear una estructura jerárquica para análisis
jerarquia_anidada = jerarquia_df.groupby(['Departamento', 'Equipo'])['Miembro'].apply(list).reset_index()
print("\nEstructura jerárquica:")
print(jerarquia_anidada)

# Transformar a formato JSON para representar la jerarquía
def crear_jerarquia_json(df):
    resultado = {}
    for _, fila in df.iterrows():
        depto = fila['Departamento']
        equipo = fila['Equipo']
        miembro = fila['Miembro']
        
        if depto not in resultado:
            resultado[depto] = {}
        
        if equipo not in resultado[depto]:
            resultado[depto][equipo] = []
        
        resultado[depto][equipo].append(miembro)
    
    return resultado

jerarquia_json = crear_jerarquia_json(jerarquia_df)
print("\nJerarquía en formato JSON:")
print(jerarquia_json)
separator_line()

# Datos complejos: registros médicos (simulados)
registros_medicos = pd.Series([
    "Paciente: #12345 (M/45) | Diagnóstico: Hipertensión (CIE-10: I10) | Medicación: Enalapril 10mg/día",
    "Paciente: #67890 (F/38) | Diagnóstico: Diabetes tipo 2 (CIE-10: E11.9) | Medicación: Metformina 850mg/12h",
    "Paciente: #24680 (M/62) | Diagnóstico: Artritis reumatoide (CIE-10: M05.9) | Medicación: Prednisona 5mg/día, Metotrexato 15mg/semana"
])

# Extraer información básica
info_basica = registros_medicos.str.extract(
    r'Paciente: #(\d+) \(([MF])/(\d+)\) \| Diagnóstico: (.*?) \(CIE-10: (.*?)\) \| Medicación: (.*)'
)
info_basica.columns = ['ID', 'Sexo', 'Edad', 'Diagnóstico', 'Código_CIE', 'Medicación']
info_basica['Edad'] = info_basica['Edad'].astype(int)

print("Información básica extraída:")
print(info_basica)

# Procesar la medicación (puede contener múltiples medicamentos)
def procesar_medicacion(med_texto):
    medicamentos = [m.strip() for m in med_texto.split(',')]
    resultado = []
    
    for med in medicamentos:
        # Extraer nombre y dosis
        partes = med.split(' ', 1)
        if len(partes) == 2:
            nombre, dosis = partes
            resultado.append({'nombre': nombre, 'dosis': dosis})
    
    return resultado

# Aplicar procesamiento de medicación
info_basica['Medicamentos'] = info_basica['Medicación'].apply(procesar_medicacion)

# Expandir la lista de medicamentos en filas separadas
registros_expandidos = info_basica.explode('Medicamentos')

# Extraer las columnas de los diccionarios de medicamentos
medicamentos_df = pd.json_normalize(registros_expandidos['Medicamentos'])
registros_expandidos = pd.concat([
    registros_expandidos.drop('Medicamentos', axis=1).reset_index(drop=True),
    medicamentos_df.reset_index(drop=True)
], axis=1)

# Eliminar la columna original de medicación
registros_expandidos = registros_expandidos.drop('Medicación', axis=1)

print("\nRegistros expandidos con medicamentos:")
print(registros_expandidos)

separator_line()


