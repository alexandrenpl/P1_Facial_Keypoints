## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # --- Arquitetura Final e Estável ---
        # 4 camadas convolucionais, inspirada no projeto de referência
        
        # Input: 224x224
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) # 224 -> 220
        self.pool1 = nn.MaxPool2d(2, 2) # -> 110
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 110 -> 108
        self.pool2 = nn.MaxPool2d(2, 2) # -> 54
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # 54 -> 52
        self.pool3 = nn.MaxPool2d(2, 2) # -> 26
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3) # 26 -> 24
        self.pool4 = nn.MaxPool2d(2, 2) # -> 12
        self.bn4 = nn.BatchNorm2d(256)
        
        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(256 * 12 * 12, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        
        self.fc2 = nn.Linear(1000, 500)
        self.fc2_bn = nn.BatchNorm1d(500)

        self.fc3 = nn.Linear(500, 136)
        
        self.dropout_fc = nn.Dropout(p=0.4)

    def forward(self, x):
        # Passagem pelas camadas convolucionais
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Achatamento (Flatten)
        x = x.view(x.size(0), -1)
        
        # Passagem pelas camadas totalmente conectadas
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout_fc(x)
        
        x = self.fc3(x)
        
        return x

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)

