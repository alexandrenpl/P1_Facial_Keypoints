import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Convolutional blocks (GN, estável com batch pequeno) ---
        # Input: 224x224 grayscale
        self.conv1 = nn.Conv2d(1, 32, 5)                 # 224 -> 220
        self.gn1   = nn.GroupNorm(32, 32, eps=1e-4)      # GN em vez de BN
        self.pool1 = nn.MaxPool2d(2, 2)                  # 220 -> 110

        self.conv2 = nn.Conv2d(32, 64, 3)                # 110 -> 108
        self.gn2   = nn.GroupNorm(32, 64, eps=1e-4)
        self.pool2 = nn.MaxPool2d(2, 2)                  # 108 -> 54

        self.conv3 = nn.Conv2d(64, 128, 3)               # 54 -> 52
        self.gn3   = nn.GroupNorm(32, 128, eps=1e-4)
        self.pool3 = nn.MaxPool2d(2, 2)                  # 52 -> 26

        self.conv4 = nn.Conv2d(128, 300, 3)              # 26 -> 24
        self.gn4   = nn.GroupNorm(30, 300, eps=1e-4)     # 30 grupos divide 300
        self.pool4 = nn.MaxPool2d(2, 2)                  # 24 -> 12

        # --- Fully connected (use LayerNorm, não há GN1d para Linear) ---
        self.fc1    = nn.Linear(300 * 12 * 12, 1100)
        self.ln1    = nn.LayerNorm(1100, eps=1e-4)

        self.fc2    = nn.Linear(1100, 550)
        self.ln2    = nn.LayerNorm(550, eps=1e-4)

        self.fc3    = nn.Linear(550, 136)  # 68*2
        self.dropout_fc = nn.Dropout(0.35)

    def forward(self, x):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))
        x = self.pool4(F.relu(self.gn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.ln1(self.fc1(x))); x = self.dropout_fc(x)
        x = F.relu(self.ln2(self.fc2(x))); x = self.dropout_fc(x)
        return self.fc3(x)
