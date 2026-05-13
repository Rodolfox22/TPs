import cv2
import numpy as np

# ------------------------------------------------------------
# 1. CONFIGURACIÓN DE PARÁMETROS
# ------------------------------------------------------------
UMBRAL_GRIS = 127  # Reducido para mejor detección
AREA_MINIMA = 1000  # Reducido para detectar figuras más pequeñas
TOLERANCIA_CUADRADO = 0.2  # Aumentado para más flexibilidad (0.8 a 1.2)
COLOR_BOX = (0, 255, 0)  # Verde
GROSOR_LINEA = 2
FUENTE_TEXTO = cv2.FONT_HERSHEY_SIMPLEX
ESCALA_TEXTO = 0.7
COLOR_TEXTO = (255, 0, 0)  # Azul
GROSOR_TEXTO = 2

# ------------------------------------------------------------
# 2. PIPELINE DE PROCESAMIENTO COMPLETO
# ------------------------------------------------------------

# Cargar la imagen
imagen_original = cv2.imread('TP1 -U5/piezas.jpg')
if imagen_original is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Crear copia para procesamiento
imagen_procesada = imagen_original.copy()

# ============================================================
# IMAGEN 1: ORIGINAL
# ============================================================
cv2.imshow('1. Imagen Original', imagen_original)

# ============================================================
# IMAGEN 2: ESCALA DE GRISES
# ============================================================
imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
cv2.imshow('2. Escala de Grises', imagen_gris)

# ============================================================
# IMAGEN 3: BINARIZACIÓN
# ============================================================
# Método 1: Umbral simple (para imágenes con buen contraste)
_, imagen_binaria1 = cv2.threshold(imagen_gris, UMBRAL_GRIS, 255, cv2.THRESH_BINARY_INV)

# Método 2: Umbral adaptativo (mejor para iluminación variable)
imagen_binaria2 = cv2.adaptiveThreshold(imagen_gris, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Probar ambos métodos y usar el que mejor funcione
# Para este ejemplo, usamos el umbral adaptativo que es más robusto
imagen_binaria = imagen_binaria2
cv2.imshow('3. Imagen Binarizada (Adaptativa)', imagen_binaria)

# Opcional: Operaciones morfológicas para limpiar ruido
kernel = np.ones((3,3), np.uint8)
imagen_binaria = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)
imagen_binaria = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

# Mostrar versión mejorada de la binarizada
cv2.imshow('3b. Imagen Binarizada (Mejorada)', imagen_binaria)

# ============================================================
# DETECCIÓN DE CONTORNOS
# ============================================================
# Usamos RETR_EXTERNAL para contornos externos o RETR_LIST para todos
contornos, jerarquia = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Contornos detectados: {len(contornos)}")

# Crear una imagen para mostrar solo los contornos
imagen_contornos = np.zeros_like(imagen_gris)
cv2.drawContours(imagen_contornos, contornos, -1, 255, 2)
cv2.imshow('3c. Contornos Detectados', imagen_contornos)

# ============================================================
# IMAGEN 4: CLASIFICACIÓN FINAL
# ============================================================
# Contadores para estadísticas
cuadrados_encontrados = 0
rectangulos_encontrados = 0

for i, contorno in enumerate(contornos):
    # Calcular el área
    area = cv2.contourArea(contorno)
    
    # Filtrar contornos muy pequeños (ruido)
    if area < AREA_MINIMA:
        print(f"Contorno {i}: Área {area:.0f} - DESCARTADO (menor a {AREA_MINIMA})")
        continue
    
    # Obtener el bounding box
    x, y, ancho, alto = cv2.boundingRect(contorno)
    
    # Evitar división por cero
    if alto == 0:
        continue
    
    # Calcular relación de aspecto
    relacion_aspecto = ancho / alto
    
    # Aproximar el polígono para verificar si es realmente un rectángulo
    epsilon = 0.02 * cv2.arcLength(contorno, True)
    vertices = cv2.approxPolyDP(contorno, epsilon, True)
    num_vertices = len(vertices)
    
    # Clasificar la forma
    # Un rectángulo/cuadrado debe tener 4 vértices aproximadamente
    if num_vertices >= 4 and num_vertices <= 6:  # Tolerancia para imperfecciones
        if abs(relacion_aspecto - 1.0) <= TOLERANCIA_CUADRADO:
            nombre_figura = "CUADRADO"
            cuadrados_encontrados += 1
            color_figura = (0, 255, 255)  # Amarillo para cuadrados
        else:
            nombre_figura = "RECTANGULO"
            rectangulos_encontrados += 1
            color_figura = (255, 0, 255)  # Magenta para rectángulos
    else:
        # Si no tiene 4 vértices, podría ser otra forma (opcional)
        nombre_figura = f"POLIGONO ({num_vertices} lados)"
        color_figura = (255, 255, 0)  # Cyan para otros
    
    # Dibujar el bounding box
    cv2.rectangle(imagen_procesada, (x, y), (x + ancho, y + alto), COLOR_BOX, GROSOR_LINEA)
    
    # Calcular posición para el texto (centrado)
    (w_texto, h_texto), _ = cv2.getTextSize(nombre_figura, FUENTE_TEXTO, ESCALA_TEXTO, GROSOR_TEXTO)
    pos_x = x + (ancho - w_texto) // 2
    pos_y = y - 10 if y - 10 > 20 else y + alto + 20
    
    # Escribir el nombre de la figura
    cv2.putText(imagen_procesada, nombre_figura, (pos_x, pos_y), 
                FUENTE_TEXTO, ESCALA_TEXTO, color_figura, GROSOR_TEXTO)
    
    # Agregar información adicional (área y relación) para debug
    info_texto = f"Area:{area:.0f} R:{relacion_aspecto:.2f}"
    cv2.putText(imagen_procesada, info_texto, (x, y + alto + 15), 
                FUENTE_TEXTO, 0.4, (200, 200, 200), 1)
    
    # Imprimir en consola
    print(f"Contorno {i}: {nombre_figura}")
    print(f"  - Área: {area:.0f} píxeles")
    print(f"  - Dimensiones: {ancho}x{alto}")
    print(f"  - Relación ancho/alto: {relacion_aspecto:.3f}")
    print(f"  - Vértices aproximados: {num_vertices}")
    print(f"  - Posición: ({x}, {y})")
    print("-" * 40)

# Agregar resumen en la imagen final
resumen = f"Total: {cuadrados_encontrados} Cuadrados, {rectangulos_encontrados} Rectangulos"
cv2.putText(imagen_procesada, resumen, (10, imagen_procesada.shape[0] - 20), 
            FUENTE_TEXTO, 0.6, (255, 255, 255), 2)

# ============================================================
# 4. MOSTRAR IMAGEN FINAL
# ============================================================
cv2.imshow('4. Clasificacion Final', imagen_procesada)

print("=" * 50)
print(f"RESUMEN FINAL:")
print(f"  - Cuadrados detectados: {cuadrados_encontrados}")
print(f"  - Rectángulos detectados: {rectangulos_encontrados}")
print(f"  - Total figuras clasificadas: {cuadrados_encontrados + rectangulos_encontrados}")
print("=" * 50)

# Esperar a que el usuario presione una tecla
cv2.waitKey(0)
cv2.destroyAllWindows()