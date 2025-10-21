'''Operaciones matemáticas básicas utilizando NumPy'''
import time
import numpy as np

# Creamos dos arrays
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Operaciones básicas
suma = a + b          # [6, 8, 10, 12]
resta = a - b         # [-4, -4, -4, -4]
multiplicacion = a * b  # [5, 12, 21, 32]
division = a / b      # [0.2, 0.33, 0.43, 0.5]
potencia = a ** 2     # [1, 4, 9, 16]

print(f'Suma: {suma}')
print(f'Resta: {resta}')
print(f'Multiplicación: {multiplicacion}')
print(f'División: {division}')
print(f'Potencia: {potencia}')

print('------------------------------------------------------------------------')
# Array y escalar
c = np.array([10, 20, 30, 40])
ESCALAR = 5

suma_escalar = c + ESCALAR      # [15, 25, 35, 45]
mult_escalar = c * ESCALAR      # [50, 100, 150, 200]
div_escalar = c / ESCALAR       # [2., 4., 6., 8.]
print(f'Suma con escalar: {suma_escalar}')
print(f'Multiplicación con escalar: {mult_escalar}')
print(f'División con escalar: {div_escalar}')
potencia_escalar = c ** ESCALAR  # [100000, 3200000000, 205891132094649, 10995116277760000]
print(f'Potencia con escalar: {potencia_escalar}')
print('------------------------------------------------------------------------')

# Normalizaciond de datos
# Datos de ejemplo (cada fila es una muestra, cada columna una característica)
datos = np.array([
    [10, 20, 30],
    [15, 25, 35],
    [20, 30, 40],
    [25, 35, 45]
])

# Calculamos media y desviación estándar por columna
medias = np.mean(datos, axis=0)  # [17.5, 27.5, 37.5]
desviaciones = np.std(datos, axis=0)  # [6.45, 6.45, 6.45]

# Normalizamos los datos usando broadcasting
datos_normalizados = (datos - medias) / desviaciones
print(datos_normalizados)
# Resultado aproximado:
# [[-1.16, -1.16, -1.16]
#  [-0.39, -0.39, -0.39]
#  [ 0.39,  0.39,  0.39]
#  [ 1.16,  1.16,  1.16]]

print('------------------------------------------------------------------------')

# Calculo de distancias
# Conjunto 1: 3 puntos en 2D
puntos_a = np.array([[1, 2], [3, 4], [5, 6]])  # Shape: (3, 2)

# Conjunto 2: 2 puntos en 2D
puntos_b = np.array([[7, 8], [9, 10]])  # Shape: (2, 2)

# Expandimos dimensiones para preparar el broadcasting
a_expandido = puntos_a[:, np.newaxis, :]  # Shape: (3, 1, 2)
b_expandido = puntos_b[np.newaxis, :, :]  # Shape: (1, 2, 2)

# Calculamos diferencias y luego distancias euclidianas
diferencias = a_expandido - b_expandido  # Shape: (3, 2, 2)
distancias = np.sqrt(np.sum(diferencias**2, axis=2))
print(distancias)
# Resultado:
# [[ 8.49, 11.31]
#  [ 5.66,  8.49]
#  [ 2.83,  5.66]]

print('------------------------------------------------------------------------')

# Operadores de comparación y broadcasting

a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 2, 2])

# Comparaciones con broadcasting
comparacion = a > b  # [False, False, True, True]

# Uso práctico: filtrado de datos
datos = np.array([15, 20, 25, 30, 35, 40])
UMBRAL = 30
filtrados = datos[datos >= UMBRAL]  # [30, 35, 40]
print(f'Comparación a > b: {comparacion}')
print(f'Datos filtrados >= {UMBRAL}: {filtrados}')
print('------------------------------------------------------------------------')

# Anatomía de una ufunc

# Creamos un array de ejemplo
arr = np.array([0, np.pi/4, np.pi/2, np.pi])

# Aplicamos una ufunc trigonométrica
senos = np.sin(arr)
print(senos)  # [0.0, 0.7071067811865475, 1.0, 0.0]

# Ufuncs matemáticas básicas

# Creamos arrays de ejemplo
a = np.array([-2, -1, 0, 1, 2])
b = np.array([1, 2, 3, 4, 5])

# Valor absoluto
abs_a = np.abs(a)  # [2, 1, 0, 1, 2]

# Raíz cuadrada (valores negativos resultan en NaN)
sqrt_b = np.sqrt(b)  # [1.0, 1.414, 1.732, 2.0, 2.236]

# Redondeo
c = np.array([1.49, 1.51, 2.49, 2.51])
redondeado = np.round(c)  # [1.0, 2.0, 2.0, 3.0]
redondeado_abajo = np.floor(c)  # [1.0, 1.0, 2.0, 2.0]
redondeado_arriba = np.ceil(c)  # [2.0, 2.0, 3.0, 3.0]

print(f'Valor absoluto de a: {abs_a}')
print(f'Raíz cuadrada de b: {sqrt_b}')
print(f'Redondeo: {redondeado}')
print(f'Redondeo hacia abajo: {redondeado_abajo}')
print(f'Redondeo hacia arriba: {redondeado_arriba}')
print('------------------------------------------------------------------------')

# Ufuncs trigonométricas

# Ángulos en radianes
angulos = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

# Funciones trigonométricas
seno = np.sin(angulos)  # [0.0, 0.5, 0.7071, 0.866, 1.0]
coseno = np.cos(angulos)  # [1.0, 0.866, 0.7071, 0.5, 0.0]
tangente = np.tan(angulos)  # [0.0, 0.577, 1.0, 1.732, ∞]

# Funciones trigonométricas inversas
valores = np.array([-1, -0.5, 0, 0.5, 1])
arcoseno = np.arcsin(valores)  # [-π/2, -π/6, 0, π/6, π/2]
arcocoseno = np.arccos(valores)  # [π, 2π/3, π/2, π/3, 0]

# Ufuncs exponenciales y logarítmicas

# Array de ejemplo
x = np.array([0, 1, 2, 3])

# Exponenciales
exp_x = np.exp(x)  # [1.0, 2.718, 7.389, 20.086]
exp2_x = np.exp2(x)  # [1.0, 2.0, 4.0, 8.0]

# Logaritmos
log_natural = np.log(exp_x)  # [0.0, 1.0, 2.0, 3.0]
log_base10 = np.log10(np.array([1, 10, 100, 1000]))  # [0.0, 1.0, 2.0, 3.0]
log_base2 = np.log2(np.array([1, 2, 4, 8]))  # [0.0, 1.0, 2.0, 3.0]

# Ufuncs binarias

# Arrays de ejemplo
a = np.array([10, 20, 30, 40])
b = np.array([2, 2, 3, 4])

# Operaciones binarias
potencia = np.power(a, b)  # [100, 400, 27000, 2560000]
modulo = np.mod(a, b)  # [0, 0, 0, 0]
maximo = np.maximum(a, b)  # [10, 20, 30, 40]
minimo = np.minimum(a, b)  # [2, 2, 3, 4]

# Ufuncs para comparaciones

# Arrays para comparar
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 2, 2])

# Comparaciones como ufuncs
igual = np.equal(a, b)  # [False, True, False, False]
mayor = np.greater(a, b)  # [False, False, True, True]
menor_igual = np.less_equal(a, b)  # [True, True, False, False]

# Útil para encontrar valores cercanos (con tolerancia)
casi_igual = np.isclose(np.array([1.0, 1.1, 1.01]), 1.0, rtol=0.1)  # [True, True, True]

# Operaciones con máscaras booleanas

# Datos de ejemplo
temperaturas = np.array([15, 22, 18, 30, 27, 12, 35, 24])

# Creamos una máscara para temperaturas altas (>25°C)
mascara_calor = np.greater(temperaturas, 25)  # [False, False, False, True, True, False, True, False]

# Filtramos usando la máscara
dias_calurosos = temperaturas[mascara_calor]  # [30, 27, 35]

# Contamos cuántos días calurosos hay
total_dias_calurosos = np.sum(mascara_calor)  # 3

# Ufuncs para procesamiento de datos

# Datos con posibles valores faltantes (NaN)
datos = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan])

# Comprobar valores NaN
mascara_nan = np.isnan(datos)  # [False, False, True, False, False, True]

# Reemplazar NaN con ceros
datos_limpios = np.nan_to_num(datos)  # [1.0, 2.0, 0.0, 4.0, 5.0, 0.0]

# O filtrar los NaN
datos_filtrados = datos[~np.isnan(datos)]  # [1.0, 2.0, 4.0, 5.0]

# Encadenamiento de ufuncs

# Array de ejemplo
x = np.array([-2, -1, 0, 1, 2])

# Encadenamos múltiples ufuncs
resultado = np.abs(np.sin(x) * np.exp(-np.square(x) / 2))
print(resultado)  # [0.142, 0.303, 0.0, 0.303, 0.142]

print('------------------------------------------------------------------------')
# Rendimiento de las ufuncs

# Creamos un array grande
TAMANIO = 10_000_000
arr = np.random.random(TAMANIO)

# Medimos tiempo con ufunc de NumPy
inicio = time.time()
resultado_numpy = np.sqrt(arr)
fin = time.time()
print(f"NumPy ufunc: {fin - inicio:.6f} segundos")

# Comparamos con Python puro (mucho más lento)
inicio = time.time()
resultado_python = [x**0.5 for x in arr]
fin = time.time()
print(f"Python puro: {fin - inicio:.6f} segundos")
print('------------------------------------------------------------------------')
# Ufuncs con argumentos adicionales

# Array con valores extremos
datos = np.array([-1000, 1, 2, 3, 1000])

# Recortar valores a un rango específico
recortados = np.clip(datos, -10, 10)  # [-10, 1, 2, 3, 10]

# División con manejo personalizado de errores
a = np.array([1, 0, 3, 0])
b = np.array([0, 0, 2, 0])

# Ignorar errores (produce infinitos y NaN)
# division1 = np.divide(a, b, out=np.zeros_like(a), where=b!=0)  # [0, 0, 1.5, 0]

print('------------------------------------------------------------------------')
# Creación de ufuncs personalizadas

# Definimos una función Python normal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Creamos un array de prueba
valores = np.array([-10, -1, 0, 1, 10])

# Aplicamos nuestra función (NumPy vectoriza automáticamente)
activaciones = sigmoid(valores)  # [~0.0, 0.269, 0.5, 0.731, ~1.0]
print(activaciones)

# Para máximo rendimiento, podríamos usar np.vectorize o numba
# import numba
# sigmoid_optimizado = numba.vectorize(sigmoid)
print('------------------------------------------------------------------------')


# Introducción a numpy.linalg
# El submódulo linalg está diseñado específicamente para operaciones matriciales avanzadas que son fundamentales en campos como la física, ingeniería, estadística y aprendizaje automático.

from numpy import linalg as LA  # Es común importarlo con un alias

# Normas vectoriales y matriciales
# Las normas son medidas de la magnitud o "tamaño" de vectores y matrices. NumPy proporciona la función norm() para calcular diferentes tipos de normas:

# Vector de ejemplo
v = np.array([3, 4])

# Norma L2 (euclidiana) - la distancia desde el origen
norma_l2 = LA.norm(v)  # 5.0 (el teorema de Pitágoras: √(3² + 4²))

# Norma L1 (suma de valores absolutos)
norma_l1 = LA.norm(v, ord=1)  # 7.0 (|3| + |4|)

# Norma infinito (valor máximo absoluto)
norma_inf = LA.norm(v, ord=np.inf)  # 4.0 (max(|3|, |4|))

# Normas para matrices
A = np.array([[1, 2], [3, 4]])
norma_frobenius = LA.norm(A)  # 5.477 (raíz cuadrada de la suma de cuadrados)


# Las normas son útiles para medir distancias, errores y para normalizar vectores.

# Solución de sistemas de ecuaciones lineales Uno de los problemas fundamentales en álgebra lineal es resolver sistemas de ecuaciones lineales de la forma Ax = b. NumPy proporciona la función solve() para este propósito:

# Sistema de ecuaciones:
# 2x + y = 5
# 3x + 2y = 8

# Matriz de coeficientes A
A = np.array([[2, 1], 
              [3, 2]])

# Vector de términos independientes b
b = np.array([5, 8])

# Resolvemos el sistema Ax = b
x = LA.solve(A, b)
print(x)  # [1. 3.] (x=1, y=3)

# Verificamos la solución
np.allclose(np.dot(A, x), b)  # True

print('------------------------------------------------------------------------')

# Esta función es mucho más eficiente y numéricamente estable que intentar calcular la inversa explícitamente.

# Descomposición en valores singulares (SVD) La descomposición en valores singulares es una factorización que descompone una matriz en tres componentes: A = U·Σ·Vᵀ. Es una herramienta fundamental en análisis de datos, compresión de imágenes y reducción de dimensionalidad.

# Matriz de ejemplo
A = np.array([[1, 2], [3, 4], [5, 6]])

# Descomposición SVD completa
U, s, Vt = LA.svd(A, full_matrices=True)

# U: matriz unitaria izquierda (3x3)
# s: valores singulares (2)
# Vt: matriz unitaria derecha transpuesta (2x2)

print(f"Forma de U: {U.shape}")  # (3, 3)
print(f"Valores singulares: {s}")  # [9.52 0.51]
print(f"Forma de Vt: {Vt.shape}")  # (2, 2)

# Reconstrucción de la matriz original
# Primero creamos Sigma como matriz diagonal
Sigma = np.zeros((3, 2))
Sigma[:2, :2] = np.diag(s)

# Reconstruimos A = U·Σ·Vᵀ
A_reconstruida = U @ Sigma @ Vt
print(np.allclose(A, A_reconstruida))  # True
print('------------------------------------------------------------------------')

# Autovalores y autovectores
# Los autovalores y autovectores son conceptos fundamentales en álgebra lineal con aplicaciones en física, ingeniería y aprendizaje automático.

# Matriz cuadrada de ejemplo
A = np.array([[4, -2], 
              [1, 1]])

# Calculamos autovalores y autovectores
autovalores, autovectores = LA.eig(A)

print(f"Autovalores: {autovalores}")  # [3. 2.]
print(f"Autovectores (por columnas):\n{autovectores}")
# [[0.89 -0.71]
#  [0.45  0.71]]

# Verificamos que Av = λv para el primer autovalor/autovector
v1 = autovectores[:, 0]
lambda1 = autovalores[0]
print(np.allclose(A @ v1, lambda1 * v1))  # True

print('------------------------------------------------------------------------')

# Descomposición de Cholesky
# Para matrices simétricas definidas positivas, la descomposición de Cholesky proporciona una factorización de la forma A = L·Lᵀ, donde L es una matriz triangular inferior:

# Matriz simétrica definida positiva
A = np.array([[4, 2], 
              [2, 5]])

# Descomposición de Cholesky
L = LA.cholesky(A)

print("Matriz triangular inferior L:")
print(L)
# [[2.   0.  ]
#  [1.   2.  ]]

# Verificamos que A = L·Lᵀ
print(np.allclose(A, L @ L.T))  # True
print('------------------------------------------------------------------------')

# Cálculo de determinantes
# El determinante de una matriz cuadrada es un valor escalar que proporciona información sobre si la matriz es invertible y cómo las transformaciones lineales cambian el volumen:

# Matrices de ejemplo
A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 0], [0, 1]])  # Matriz identidad
C = np.array([[1, 2], [2, 4]])  # Matriz singular

# Calculamos determinantes
det_A = LA.det(A)
det_B = LA.det(B)
det_C = LA.det(C)

print(f"det(A) = {det_A}")  # -2.0
print(f"det(B) = {det_B}")  # 1.0
print(f"det(C) = {det_C}")  # 0.0 (matriz singular)

print('------------------------------------------------------------------------')

# Cálculo de la inversa de una matriz
# La matriz inversa A⁻¹ de una matriz A cumple que A·A⁻¹ = A⁻¹·A = I (matriz identidad):

# Matriz invertible
A = np.array([[4, 7], [2, 6]])

# Calculamos la inversa
A_inv = LA.inv(A)

print("Matriz inversa:")
print(A_inv)
# [[ 0.6  -0.7 ]
#  [-0.2   0.4 ]]

# Verificamos que A·A⁻¹ = I
producto = A @ A_inv
print(np.allclose(producto, np.eye(2)))  # True

print('------------------------------------------------------------------------')

# Cálculo del rango de una matriz
# El rango de una matriz es el número de filas o columnas linealmente independientes:

# Matrices de ejemplo
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Rango 2
B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Rango 3 (identidad)

# Calculamos rangos
rango_A = LA.matrix_rank(A)
rango_B = LA.matrix_rank(B)

print(f"Rango de A: {rango_A}")  # 2
print(f"Rango de B: {rango_B}")  # 3
print('------------------------------------------------------------------------')

# Descomposición QR
# La descomposición QR factoriza una matriz A en el producto de una matriz ortogonal Q y una matriz triangular superior R:

# Matriz de ejemplo
A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])

# Descomposición QR
Q, R = LA.qr(A)

print("Matriz ortogonal Q:")
print(Q)
# [[-0.857  0.394  0.331]
#  [-0.429 -0.903  0.034]
#  [ 0.286 -0.176  0.943]]

print("Matriz triangular superior R:")
print(R)
# [[-14.    -21.     14.   ]
#  [  0.    -175.    -70.   ]
#  [  0.      0.     -35.   ]]

# Verificamos que A = Q·R
print(np.allclose(A, Q @ R))  # True
