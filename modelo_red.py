import torch
import torch.nn as nn
import torch.nn.functional as F

class SolarNet(nn.Module):
    def __init__(self):
        super(SolarNet, self).__init__()

        # Bloques convolucionales
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Capa totalmente conectada
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # después de 3 bloques conv+pool → 64x64 → 32x32 → 16x16 → 8x8
        self.fc2 = nn.Linear(128, 2)  # salida (x, y)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x64 → 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 → 16x16
        x = self.pool(F.relu(self.conv3(x)))  # 16x16 → 8x8
        x = x.view(-1, 64 * 8 * 8)  # a vector plano
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # salida directa sin activación → regresión
        return x

# Prueba rápida
if __name__ == "__main__":
    modelo = SolarNet()
    dummy_input = torch.randn(1, 3, 64, 64)  # 1 imagen RGB de 64x64
    output = modelo(dummy_input)
    print("Salida (x, y):", output)
