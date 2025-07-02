# Contenido para tu archivo: Main.py (CORREGIDO)

import cv2
import os
from datetime import datetime
import time
import csv
# CORRECCIÓN: Importamos la función 'sun_position' (en minúsculas) de tu archivo local
from sun_position import sun_position

# --- CONFIGURACIÓN PRINCIPAL ---
carpeta_fotos = "images"
carpeta_metadatos = "Metadatos"
cam_port = 0
capture_interval = 5

# Ubicación para Los Robles La Paz, Cesar, Colombia
LATITUD = 10.38
LONGITUD = -73.48
# El huso horario de Colombia es GMT-5. El centro del huso es 5 * 15 = 75° W
TIME_ZONE_CENTER = -75.0

# --- INICIALIZACIÓN Y VERIFICACIÓN ---
os.makedirs(carpeta_fotos, exist_ok=True)
os.makedirs(carpeta_metadatos, exist_ok=True)
ruta_csv = os.path.join(carpeta_metadatos, "posiciones_sol.csv")

cap = cv2.VideoCapture(cam_port)
if not cap.isOpened():
    print(f"Error: No se pudo abrir la cámara en el puerto {cam_port}.")
    exit()

print("="*60)
print("Iniciando sistema de captura y seguimiento solar (v3 - Corregido).")
print(f"Ubicación: Los Robles La Paz (Lat: {LATITUD}, Lon: {LONGITUD})")
print(f"Huso Horario: GMT-5 (Centro en {TIME_ZONE_CENTER}° W)")
print("\n>>> PRESIONA LA TECLA 'q' SOBRE LA VENTANA DE LA CÁMARA PARA SALIR. <<<")
print("="*60)

last_capture_time = time.time()

# --- BUCLE PRINCIPAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Se perdió la conexión con la cámara.")
        break
    
    cv2.imshow('Camara en Vivo - Presiona Q para salir', frame)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        
        t_captura = datetime.now()
        
        # 1. CALCULAR LA POSICIÓN DEL SOL PRIMERO
        # CORRECCIÓN: Llamamos a la función con todos los parámetros necesarios
        # y desempaquetamos los 3 valores que retorna (x, y, mask)
        x, y, _ = sun_position(t_captura, LATITUD, LONGITUD, TIME_ZONE_CENTER)
        print(f"[{t_captura.strftime('%H:%M:%S')}] Posición en imagen calculada: Coordenada (x,y) = ({x},{y})")

        # 2. CREAR EL NOMBRE DEL ARCHIVO CON LAS COORDENADAS
        nombre_archivo_img = f"{t_captura.strftime('%Y-%m-%d_%H-%M-%S')}_x{int(x)}_y{int(y)}.png"
        ruta_completa_img = os.path.join(carpeta_fotos, nombre_archivo_img)
        
        # 3. GUARDAR LA IMAGEN REAL DE LA CÁMARA
        cv2.imwrite(ruta_completa_img, frame)
        print(f"    -> Foto guardada: {nombre_archivo_img}")

        # 4. GUARDAR LOS METADATOS EN EL ARCHIVO CSV
        guardar_header = not os.path.exists(ruta_csv)
        with open(ruta_csv, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if guardar_header:
                # Usamos los nombres de las variables que devuelve tu función (azimuth, zenith)
                # aunque en tu código se llaman x, y, representan lo mismo.
                writer.writerow(["fecha_hora", "imagen_coord_x", "imagen_coord_y"])
            writer.writerow([t_captura.strftime("%Y-%m-%d %H:%M:%S"), f"{x}", f"{y}"])
        
        # 5. REINICIAR EL TEMPORIZADOR
        last_capture_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nCerrando programa...")
        break

# --- LIMPIEZA FINAL ---
cap.release()
cv2.destroyAllWindows()
print("Recursos liberados. Programa finalizado.")