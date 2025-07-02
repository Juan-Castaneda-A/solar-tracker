# -----------------------------------------------------------------------------
# main.py - SISTEMA DE CONTROL HÍBRIDO PARA SEGUIDOR SOLAR
# Versión con Modo de Prueba para depuración en casa.
# -----------------------------------------------------------------------------
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import time
import numpy as np
import os # Importamos 'os' para manejar rutas de archivos

# --- Importaciones de tus módulos locales ---
from sun_position import sun_position
from modelo_red import SolarNet

# --- Intentamos importar la librería de GPIO, pero no fallamos si no existe (para pruebas en PC) ---
try:
    from gpiozero import Servo
    from gpiozero.pins.pigpio import PiGPIOFactory
    RPI_MODE = True
except ImportError:
    print("[ADVERTENCIA] No se encontró la librería 'gpiozero'. El control de servos estará desactivado.")
    RPI_MODE = False

# -----------------------------------------------------------------------------
# --- CONFIGURACIÓN PRINCIPAL (AJUSTA ESTOS VALORES) ---
# -----------------------------------------------------------------------------

# --- MODO DE PRUEBA CON IMAGEN FIJA ---
# Pon en True para usar una imagen de prueba y desactivar la cámara/servos.
# Pon en False para usar la cámara en vivo en la Raspberry Pi.
USE_IMAGE_MODE = True

# Si USE_IMAGE_MODE es True, especifica la ruta de la imagen que quieres probar.
# El script extraerá la fecha y hora del nombre del archivo para el modelo físico.
TEST_IMAGE_PATH = "images/2025-06-24_15-28-03_x27_y41.png" # Ejemplo con nubes
# TEST_IMAGE_PATH = "images/2025-06-17_11-03-17_x22_y22.png" # Ejemplo con sol visible

# --- Configuración Geográfica ---
LATITUD = 10.38
LONGITUD = -73.48
TIME_ZONE_CENTER = -75.0

# --- Configuración de Hardware (Solo para modo en vivo) ---
CAM_PORT = 0
CAPTURE_INTERVAL = 10
PAN_SERVO_PIN = 17
TILT_SERVO_PIN = 18

# --- Mapeo de Coordenadas a Ángulos ---
IMG_X_MIN, IMG_X_MAX = 0, 63
IMG_Y_MIN, IMG_Y_MAX = 0, 63
PAN_ANGLE_MIN, PAN_ANGLE_MAX = -90, 90
TILT_ANGLE_MIN, TILT_ANGLE_MAX = -90, 90

# --- Configuración del Modelo Híbrido ---
MODELO_ENTRENADO_PATH = "modelo_entrenado.pth"
MAX_DISTANCE_THRESHOLD = 10.0

# -----------------------------------------------------------------------------
# --- LÓGICA PRINCIPAL ---
# -----------------------------------------------------------------------------

# --- Función para mapear valores (sin cambios) ---
def map_value(value, in_min, in_max, out_min, out_max):
    if in_max == in_min: return out_min
    normalized_value = (value - in_min) / (in_max - in_min)
    mapped_value = normalized_value * (out_max - out_min) + out_min
    return max(min(mapped_value, out_max), out_min)

# --- Cargar Modelo Autónomo (común para ambos modos) ---
try:
    modelo_autonomo = SolarNet()
    modelo_autonomo.load_state_dict(torch.load(MODELO_ENTRENADO_PATH))
    modelo_autonomo.eval()
    print("[OK] Modelo autónomo 'modelo_entrenado.pth' cargado exitosamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo entrenado: {e}")
    exit()

# --- Transformaciones de imagen (común para ambos modos) ---
transformaciones = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# --- EJECUCIÓN SEGÚN EL MODO SELECCIONADO ---
if USE_IMAGE_MODE:
    # ------------------- MODO PRUEBA CON IMAGEN FIJA -------------------
    print("\n" + "="*60)
    print("--- MODO DE PRUEBA CON IMAGEN FIJA ACTIVADO ---")
    print(f"--- Usando imagen: {TEST_IMAGE_PATH} ---")
    print("="*60 + "\n")

    # Cargar la imagen de prueba desde el archivo
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        print(f"[ERROR] No se pudo cargar la imagen de prueba. Verifica la ruta: {TEST_IMAGE_PATH}")
        exit()

    # Extraer la fecha y hora del nombre del archivo para el modelo físico
    try:
        filename = os.path.basename(TEST_IMAGE_PATH)
        datetime_str = filename.split('_x')[0]
        t_captura = datetime.strptime(datetime_str, '%Y-%m-%d_%H-%M-%S')
        print(f"Fecha y hora extraídas del archivo: {t_captura}")
    except Exception as e:
        print(f"[ERROR] El nombre del archivo no tiene el formato esperado. Usando hora actual. Error: {e}")
        t_captura = datetime.now()

    # 1. PREDICCIÓN DEL MODELO FÍSICO
    x_fisico, y_fisico, _ = sun_position(t_captura, LATITUD, LONGITUD, TIME_ZONE_CENTER)
    print(f"  > Modelo Físico      (Teórico):   (x={x_fisico}, y={y_fisico})")

    # 2. PREDICCIÓN DEL MODELO AUTÓNOMO
    imagen_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imagen_tensor = transformaciones(imagen_pil).unsqueeze(0)
    with torch.no_grad():
        prediccion = modelo_autonomo(imagen_tensor)
        x_autonomo, y_autonomo = prediccion[0].tolist()
    print(f"  > Modelo Autónomo    (Visual):    (x={x_autonomo:.1f}, y={y_autonomo:.1f})")

    # 3. LÓGICA HÍBRIDA: DECISIÓN Y SELECCIÓN [cite: 18]
    distancia = np.sqrt((x_fisico - x_autonomo)**2 + (y_fisico - y_autonomo)**2)
    print(f"  > Distancia entre modelos: {distancia:.2f} píxeles")

    if distancia < MAX_DISTANCE_THRESHOLD:
        x_final, y_final = x_autonomo, y_autonomo
        active_model = "AUTÓNOMO (Cielo Despejado)"
    else:
        x_final, y_final = x_fisico, y_fisico
        active_model = "FÍSICO (Cielo Obstruido)"

    print(f"  >> MODELO ACTIVO: {active_model} <<")
    print(f"  >> Posición Objetivo Final: (x={x_final:.1f}, y={y_final:.1f})")

    # 4. CÁLCULO DE ÁNGULOS (SIN MOVER SERVOS)
    pan_angle = map_value(x_final, IMG_X_MIN, IMG_X_MAX, PAN_ANGLE_MIN, PAN_ANGLE_MAX)
    tilt_angle = map_value(y_final, IMG_Y_MIN, IMG_Y_MAX, TILT_ANGLE_MIN, TILT_ANGLE_MAX)
    print(f"  >> Ángulos calculados -> PAN: {pan_angle:.1f}°, TILT: {tilt_angle:.1f}°")

    # 5. Visualizar resultado
    # Dibuja círculos para ver las predicciones en la imagen
    frame_resized = cv2.resize(frame, (256, 256))
    scale = 256 / 64 # Escala para dibujar en la imagen más grande
    # Círculo azul para el modelo físico
    cv2.circle(frame_resized, (int(x_fisico * scale), int(y_fisico * scale)), int(5*scale), (255, 0, 0), 2)
    # Círculo rojo para el modelo autónomo
    cv2.circle(frame_resized, (int(x_autonomo * scale), int(y_autonomo * scale)), int(5*scale), (0, 0, 255), 2)
     # Círculo verde para la decisión final
    cv2.circle(frame_resized, (int(x_final * scale), int(y_final * scale)), int(6*scale), (0, 255, 0), 2)
    
    cv2.imshow(f'Resultado - Modelo Activo: {active_model}', frame_resized)
    print("\nMostrando resultado visual. Presiona cualquier tecla en la ventana de la imagen para salir.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # ------------------- MODO EN VIVO PARA RASPBERRY PI -------------------
    print("\n" + "="*60)
    print("--- MODO EN VIVO PARA RASPBERRY PI ACTIVADO ---")
    print("="*60 + "\n")

    if not RPI_MODE:
        print("[ERROR] Librerías de Raspberry Pi no encontradas. No se puede continuar en modo en vivo.")
        exit()

    # --- Inicializar Cámara ---
    cap = cv2.VideoCapture(CAM_PORT)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara en el puerto {CAM_PORT}.")
        exit()
    print(f"[OK] Cámara inicializada en el puerto {CAM_PORT}.")

    # --- Inicializar Servomotores ---
    factory = PiGPIOFactory()
    try:
        pan_servo = Servo(PAN_SERVO_PIN, pin_factory=factory)
        tilt_servo = Servo(TILT_SERVO_PIN, pin_factory=factory)
        pan_servo.value = map_value(0, -90, 90, -1, 1)
        tilt_servo.value = map_value(0, -90, 90, -1, 1)
        print(f"[OK] Servos inicializados en pines GPIO (PAN:{PAN_SERVO_PIN}, TILT:{TILT_SERVO_PIN}).")
    except Exception as e:
        print(f"[ERROR] No se pudieron inicializar los servos. Error: {e}")
        exit()
    
    print("\n>>> INICIANDO BUCLE DE OPERACIÓN. PRESIONA 'q' PARA SALIR. <<<\n")
    last_capture_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- Aquí iría la lógica del bucle en vivo, que es la misma que la del modo de imagen ---
            # (Se omite por brevedad, pero es el mismo bloque de predicción y decisión)
            # Para una versión final, se podría encapsular la lógica de predicción en una función
            # para no repetir código.

            current_time = time.time()
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                # La lógica completa de predicción, decisión y control de servos iría aquí...
                print(f"--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
                t_captura = datetime.now()
                x_fisico, y_fisico, _ = sun_position(t_captura, LATITUD, LONGITUD, TIME_ZONE_CENTER)
                print(f"  > Modelo Físico      (Teórico):   (x={x_fisico}, y={y_fisico})")

                imagen_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imagen_tensor = transformaciones(imagen_pil).unsqueeze(0)
                with torch.no_grad():
                    prediccion = modelo_autonomo(imagen_tensor)
                    x_autonomo, y_autonomo = prediccion[0].tolist()
                print(f"  > Modelo Autónomo    (Visual):    (x={x_autonomo:.1f}, y={y_autonomo:.1f})")

                distancia = np.sqrt((x_fisico - x_autonomo)**2 + (y_fisico - y_autonomo)**2)
                print(f"  > Distancia entre modelos: {distancia:.2f} píxeles")

                if distancia < MAX_DISTANCE_THRESHOLD:
                    x_final, y_final = x_autonomo, y_autonomo
                    active_model = "AUTÓNOMO (Cielo Despejado)"
                else:
                    x_final, y_final = x_fisico, y_fisico
                    active_model = "FÍSICO (Cielo Obstruido)"
                
                print(f"  >> MODELO ACTIVO: {active_model} <<")

                pan_angle = map_value(x_final, IMG_X_MIN, IMG_X_MAX, PAN_ANGLE_MIN, PAN_ANGLE_MAX)
                tilt_angle = map_value(y_final, IMG_Y_MIN, IMG_Y_MAX, TILT_ANGLE_MIN, TILT_ANGLE_MAX)
                
                pan_servo.value = map_value(pan_angle, PAN_ANGLE_MIN, PAN_ANGLE_MAX, -1, 1)
                tilt_servo.value = map_value(tilt_angle, TILT_ANGLE_MIN, TILT_ANGLE_MAX, -1, 1)

                print(f"  >> Moviendo Servos a -> Ángulo PAN: {pan_angle:.1f}°, Ángulo TILT: {tilt_angle:.1f}°")
                print("-" * 40 + "\n")
                
                last_capture_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Liberando recursos...")
        cap.release()
        cv2.destroyAllWindows()
        if RPI_MODE:
            pan_servo.detach()
            tilt_servo.detach()
        print("Recursos liberados. Programa finalizado.")