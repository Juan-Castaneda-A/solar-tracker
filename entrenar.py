import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from modelo_red import SolarNet
from dataset import SolarDataset  

# 1. Parámetros
ruta_imagenes = "images"  # carpeta donde guardas las imágenes 1080p
archivo_csv = "Metadatos/posiciones_sol.csv"  
tamaño_lote = 32
epocas = 20
lr = 0.001

# 2. Transformaciones
transformaciones = transforms.Compose([
    transforms.Resize((64, 64)),  # Aseguramos que todo se reduzca a 64x64
    transforms.ToTensor(),        # Convierte a tensor [0, 1]
])

# 3. Dataset y DataLoader
dataset = SolarDataset(csv_file=archivo_csv, image_dir=ruta_imagenes, transform=transformaciones)
loader = DataLoader(dataset, batch_size=tamaño_lote, shuffle=True)

# 4. Modelo
modelo = SolarNet()
criterio = nn.MSELoss()  # porque predecimos coordenadas (regresión)
optimizador = optim.Adam(modelo.parameters(), lr=lr)

# 5. Entrenamiento
for epoca in range(epocas):
    modelo.train()
    total_loss = 0
    for imagenes, coordenadas in loader:
        salida = modelo(imagenes)
        loss = criterio(salida, coordenadas.float())

        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        total_loss += loss.item()
    
    print(f"Época {epoca+1}/{epocas} - Pérdida: {total_loss:.4f}")

# 6. Guardar modelo
torch.save(modelo.state_dict(), "modelo_entrenado.pth")
print(" Modelo entrenado y guardado como 'modelo_entrenado.pth'")
