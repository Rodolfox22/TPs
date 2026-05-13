import cv2
import numpy as np
import math

# ------------------------------------------------------------
# PARÁMETROS
# ------------------------------------------------------------
AREA_MINIMA = 500
UMBRAL_CIRCULARIDAD = 0.80   # Cercano a 1 = más circular
TOLERANCIA_BBOX = 0.15       # Relación ancho/alto del bounding box

# ------------------------------------------------------------
# PIPELINE DE PROCESAMIENTO
# ------------------------------------------------------------

# 1. Imagen original
imagen_original = cv2.imread('TP1 -U5/piezas.jpg')
if imagen_original is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

cv2.imshow('1. Imagen Original', imagen_original)

# 2. Escala de grises
imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
cv2.imshow('2. Escala de Grises', imagen_gris)

# 3. Binarización
_, imagen_binaria = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('3. Imagen Binarizada', imagen_binaria)

# 4. Detección de contornos
contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. Clasificación y dibujo
imagen_resultado = imagen_original.copy()

for contorno in contornos:
    area = cv2.contourArea(contorno)

    # Filtrar contornos pequeños (ruido)
    if area < AREA_MINIMA:
        continue

    # Calcular circularidad: 4 * pi * área / perímetro²
    perimetro = cv2.arcLength(contorno, True)
    if perimetro == 0:
        continue
    circularidad = (4 * math.pi * area) / (perimetro ** 2)

    # Obtener bounding box y relación ancho/alto
    x, y, ancho, alto = cv2.boundingRect(contorno)
    relacion_aspecto = ancho / alto

    # Criterio circular:
    # - Circularidad cercana a 1
    # - Bounding box aproximadamente cuadrado (ancho ≈ alto)
    bbox_cuadrada = abs(relacion_aspecto - 1.0) <= TOLERANCIA_BBOX
    es_circulo = circularidad >= UMBRAL_CIRCULARIDAD and bbox_cuadrada

    if not es_circulo:
        continue

    # Dibujar bounding box y etiqueta
    cv2.rectangle(imagen_resultado, (x, y), (x + ancho, y + alto), (0, 165, 255), 2)  # Naranja
    cv2.putText(imagen_resultado, f"CIRCULO ({circularidad:.2f})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

cv2.imshow('4. Clasificacion Final - Circulos', imagen_resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()