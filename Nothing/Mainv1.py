from datetime import datetime
from sun_positionv1 import sun_position
import matplotlib.pyplot as plt
import os
import csv

t = datetime.now()
print(f"Fecha y hora actual: {t}")

# Obtener la posición del sol
x, y, mask = sun_position(t)

print(f"Posición del sol: x={x}, y={y}")

# Asegurar que la carpeta 'Metadatos' exista
carpeta = "Metadatos"
os.makedirs(carpeta, exist_ok=True)

# Ruta completa al archivo CSV dentro de la carpeta
filename = os.path.join(carpeta, "posiciones_sol.csv")
guardar_header = not os.path.exists(filename)

# Guardar los datos
with open(filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    if guardar_header:
        writer.writerow(["fecha_hora", "x", "y"])
    writer.writerow([t.strftime("%Y-%m-%d %H:%M:%S"), x, y])

print(f"Coordenadas guardadas en {filename}")

# Mostrar la máscara (imagen 64x64 con el sol dibujado)
plt.imshow(mask)
plt.title(f"Sol en ({x}, {y})")
plt.show()
