'''Análisis del consumo de memoria de imágenes en diferentes formatos de datos'''
import numpy as np

# Simulamos una imagen en escala de grises (valores 0-255)
# Usando diferentes tipos de datos
altura, anchura = 1080, 1920  # Resolución Full HD

# Creamos la misma imagen con diferentes tipos
img_uint8 = np.zeros((altura, anchura), dtype=np.uint8)
img_float32 = np.zeros((altura, anchura), dtype=np.float32)
img_float64 = np.zeros((altura, anchura), dtype=np.float64)

# Comparamos el consumo de memoria
print("Imagen uint8 (estándar para imágenes):")
print(f"  Forma: {img_uint8.shape}")
print(f"  Tipo: {img_uint8.dtype}")
print(f"  Memoria: {img_uint8.nbytes / (1024*1024):.2f} MB")

print("\nImagen float32 (común en procesamiento):")
print(f"  Forma: {img_float32.shape}")
print(f"  Tipo: {img_float32.dtype}")
print(f"  Memoria: {img_float32.nbytes / (1024*1024):.2f} MB")

print("\nImagen float64 (alta precisión):")
print(f"  Forma: {img_float64.shape}")
print(f"  Tipo: {img_float64.dtype}")
print(f"  Memoria: {img_float64.nbytes / (1024*1024):.2f} MB")
