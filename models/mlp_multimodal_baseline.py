import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ==========================================
# MODELO MULTIMODAL CORREGIDO
# ==========================================

class CNNMultimodalBaseline(nn.Module):
    def __init__(self, tab_size):
        super().__init__()

        # Rama de imagen (CNN pequeña)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=19, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15x15 → 7x7

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7 → 3x3
        )

        self.flatten = nn.Flatten()

        # tamaño salida CNN (16 canales * 3 * 3)
        cnn_out_size = 16 * 3 * 3

        # Rama tabular
        self.tabular = nn.Sequential(
            nn.Linear(tab_size, 16),
            nn.ReLU()
        )

        # Fusión
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_size + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_img, x_tab):
        x_img = self.cnn(x_img)
        x_img = self.flatten(x_img)

        x_tab = self.tabular(x_tab)

        x = torch.cat((x_img, x_tab), dim=1)

        return self.classifier(x)


# ==========================================
# TRAINER CORREGIDO
# ==========================================

class CNNMultimodalTrainer:
    def __init__(self, epochs=20, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.lr = lr

        self.model = None
        self.criterion = nn.BCELoss()
        self.optimizer = None

    def train(self, X_img, X_tab, y):

        X_img = torch.tensor(X_img, dtype=torch.float32).to(self.device)
        X_tab = torch.tensor(X_tab, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        tab_size = X_tab.shape[1]

        self.model = CNNMultimodalBaseline(tab_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()

        for epoch in range(self.epochs):
            outputs = self.model(X_img, X_tab)
            loss = self.criterion(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"[CNN-MM] Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.4f}")

    def evaluate(self, X_img, X_tab, y):
        self.model.eval()

        X_img = torch.tensor(X_img, dtype=torch.float32).to(self.device)
        X_tab = torch.tensor(X_tab, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(X_img, X_tab).cpu().numpy().flatten()

        roc = roc_auc_score(y, preds)
        acc = accuracy_score(y, (preds > 0.5).astype(int))

        print(f"[CNN MULTIMODAL BASELINE] ROC-AUC: {roc:.4f}")
        print(f"[CNN MULTIMODAL BASELINE] Accuracy: {acc:.4f}")

        return {"roc_auc": roc, "accuracy": acc}