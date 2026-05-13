import cv2
import os

CARPETA_CARAS = 'caras/'
CLASIFICADOR = 'haarcascade_frontalface_default.xml'


clasificador = cv2.CascadeClassifier(CLASIFICADOR)

# Implementar la función para capturar una foto desde la cámara
def capturar_foto(nombre_archivo):
    cap = cv2.VideoCapture(0)
    print("Presioná 'ESC' para capturar imagen.")
    while True:
        ret, frame = cap.read()
        cv2.imshow('Captura', frame)
        if cv2.waitKey(1) == 27:
            # COMPLETAR: guardar la imagen capturada en nombre_archivo
            break
    cap.release()
    cv2.destroyAllWindows()
    return nombre_archivo


# Registrar usuario tomando su foto y recortando su rostro
def registrar_usuario(nombre_usuario):
    imagen_path = capturar_foto(f"{nombre_usuario}.jpg")
    img = cv2.imread(imagen_path)
    img_byn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    caras = clasificador.detectMultiScale(img_byn)

    if len(caras) == 0:
        print("No se detectó ningún rostro.")
        return


    # COMPLETAR: Extraer la primera cara detectada y recortarla de la imagen original
    # COMPLETAR: Guardar la imagen recortada en la carpeta 'caras/' con el nombre del usuario
    # COMPLETAR: Borrar la imagen completa capturada
    print("Registro facial exitoso.")


def login_usuario(nombre_usuario):
    ruta_registro = os.path.join(CARPETA_CARAS, f"{nombre_usuario}.jpg")
    if not os.path.exists(ruta_registro):
        print("Usuario no registrado.")
        return

    imagen_path = capturar_foto(f"{nombre_usuario}_login.jpg")
    img = cv2.imread(imagen_path)
    img_byn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    caras = clasificador.detectMultiScale(img_byn)

    if len(caras) == 0:
        print("No se detectó ningún rostro.")
        os.remove(imagen_path)
        return

    cara_registrada = cv2.imread(ruta_registro, 0)

    for (x, y, w, h) in caras:
        cara_actual = img_byn[y:y+h, x:x+w]


        try:
            # COMPLETAR: Redimensionar la imagen registrada para que tenga las mismas dimensiones que la actual
        except:
            print("Error redimensionando imagen. Intenta nuevamente.")
            os.remove(imagen_path)
            return

        # Comparar la cara actual con la registrada usando matchTemplate
        # COMPLETAR: Usar cv2.matchTemplate con el método TM_SQDIFF_NORMED
        # COMPLETAR: Obtener el valor mínimo de similitud (min_val)
        # COMPLETAR: Imprimir el valor de similitud obtenido


        # COMPLETAR: Decidir si se concede o no el acceso, comparando min_val con un umbral = 0.2
        # COMPLETAR: Borrar la imagen temporal capturada

# Menú principal
def main():
    while True:
        print("\n--- SISTEMA DE LOGIN FACIAL ---")
        print("1. Registrar usuario")
        print("2. Iniciar sesión")
        print("3. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            usuario = input("Ingrese nombre de usuario: ").strip()
            # COMPLETAR: Llamar a la función para registrar usuario
        elif opcion == "2":
            usuario = input("Ingrese nombre de usuario: ").strip()
            # COMPLETAR: Llamar a la función para login
        elif opcion == "3":
            print("Saliendo del sistema.")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()
