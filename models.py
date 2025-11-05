import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # --- Blocos Convolucionais com BatchNorm ---
        # Arquitetura otimizada para ~50M parâmetros
        # Aumenta capacidade mantendo feature map 12x12
        
        # Input: 224x224 grayscale
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # 224 -> 220
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 220 -> 110
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 110 -> 108
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 108 -> 54
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # 54 -> 52
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 52 -> 26
        
        # Aumentamos última conv: 256 → 300 canais para maior capacidade
        self.conv4 = nn.Conv2d(128, 300, kernel_size=3)  # 26 -> 24
        self.bn4   = nn.BatchNorm2d(300)
        self.pool4 = nn.MaxPool2d(2, 2)  # 24 -> 12
        
        # --- Camadas Totalmente Conectadas ---
        # Feature map: 300 * 12 * 12 = 43,200 features
        self.fc1    = nn.Linear(300 * 12 * 12, 1100)
        self.fc1_bn = nn.BatchNorm1d(1100)
        
        self.fc2    = nn.Linear(1100, 550)
        self.fc2_bn = nn.BatchNorm1d(550)
        
        self.fc3    = nn.Linear(550, 136)  # 68 keypoints * 2
        
        self.dropout_fc = nn.Dropout(p=0.35)

        # Nota: inicialização será feita pelo notebook com net.apply(initialize_weights)

    def forward(self, x):
        # Conv blocks com ReLU
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten mantendo informação espacial
        x = x.view(x.size(0), -1)
        
        # FC layers com dropout
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout_fc(x)
        
        x = self.fc3(x)  # saída final sem ativação (regressão)
        
        return x


def initialize_weights(m):
    """Inicialização de pesos usando Kaiming (He) initialization"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
