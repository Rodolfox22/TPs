import cv2
import numpy as np

# ------------------------------------------------------------
# PARÁMETROS
# ------------------------------------------------------------
AREA_MINIMA = 500
TOLERANCIA_CUADRADO = 0.15

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

    # Aproximar polígono y verificar que sea cuadrilátero
    epsilon = 0.02 * cv2.arcLength(contorno, True)
    vertices = cv2.approxPolyDP(contorno, epsilon, True)
    if not (4 <= len(vertices) <= 6):
        continue

    # Obtener bounding box
    x, y, ancho, alto = cv2.boundingRect(contorno)
    relacion_aspecto = ancho / alto

    # Clasificar: cuadrado o rectángulo
    if abs(relacion_aspecto - 1.0) <= TOLERANCIA_CUADRADO:
        etiqueta = "CUADRADO"
        color = (0, 255, 255)   # Amarillo
    else:
        etiqueta = "RECTANGULO"
        color = (255, 0, 255)   # Magenta

    # Dibujar bounding box y nombre
    cv2.rectangle(imagen_resultado, (x, y), (x + ancho, y + alto), color, 2)
    cv2.putText(imagen_resultado, etiqueta, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imshow('4. Clasificacion Final', imagen_resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()