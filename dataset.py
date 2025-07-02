import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SolarDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []

        # Leer CSV y guardar pares (imagen, [x, y])
        with open(csv_file, "r") as file:
            # CORRECCIÓN: Usamos DictReader para leer las columnas por su nombre
            reader = csv.DictReader(file)
            for row in reader:
                # 1. Reconstruir el formato de fecha y hora como en Main.py
                fecha_hora_str = row["fecha_hora"].replace(":", "-").replace(" ", "_")
                
                # 2. Obtener las coordenadas X e Y
                # CORRECCIÓN: Usamos los nombres de columna correctos que guardó Main.py
                x = float(row["imagen_coord_x"])
                y = float(row["imagen_coord_y"])

                # 3. Construir el nombre del archivo EXACTAMENTE como fue guardado
                # CORRECCIÓN: Se añade la parte _x..._y... y se cambia la extensión a .png
                image_name = f"{fecha_hora_str}_x{int(x)}_y{int(y)}.png"
                image_path = os.path.join(self.image_dir, image_name)

                # 4. Verificar si el archivo existe y agregarlo a los datos
                if os.path.exists(image_path):
                    self.data.append((image_path, torch.tensor([x, y], dtype=torch.float32)))
                else:
                    # Esta línea es opcional, pero ayuda a depurar si aún hay problemas
                    print(f"ADVERTENCIA: No se encontró el archivo: {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, coords = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, coords

# Prueba rápida de carga
if __name__ == "__main__":
    transformaciones = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # convierte a [0, 1] y cambia forma a (C, H, W)
    ])

    dataset = SolarDataset("Metadatos/posiciones_sol.csv", "images", transform=transformaciones)
    print(f"Total de muestras: {len(dataset)}")

    imagen, coordenadas = dataset[0]
    print("Tamaño de imagen:", imagen.shape)  # (3, 64, 64)
    print("Coordenadas:", coordenadas)
