import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from modelo_red import SolarNet
import matplotlib.pyplot as plt

# 1. Cargar modelo entrenado
modelo = SolarNet()
modelo.load_state_dict(torch.load("modelo_entrenado.pth"))
modelo.eval()

# 2. Transformaciones (deben coincidir con las del entrenamiento)
transformaciones = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 3. Ruta de la imagen a predecir
ruta_imagen_original = "1212.jpg"  # <- Cambia esto por tu imagen
imagen_original = Image.open(ruta_imagen_original).convert("RGB")

# 4. Preprocesar imagen
imagen_red = transformaciones(imagen_original).unsqueeze(0)  # [1, 3, 64, 64]

# 5. Predicción
with torch.no_grad():
    salida = modelo(imagen_red)
    x_pred, y_pred = salida[0].tolist()

# 6. Mostrar resultado en imagen
imagen_marcada = imagen_original.resize((64, 64)).copy()

# 7. Mostrar con matplotlib
plt.imshow(imagen_marcada)
plt.title(f"Predicción: x={x_pred:.1f}, y={y_pred:.1f}")
plt.axis("off")
plt.show()
