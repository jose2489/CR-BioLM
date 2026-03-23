import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


class EarlyFusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


class EarlyFusionTrainer:
    def __init__(self, input_dim, epochs=50, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EarlyFusionModel(input_dim).to(self.device)
        self.epochs = epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_img, X_tab, y):
        print("[INFO] Entrenando Early Fusion...")

        # Flatten imágenes
        X_img_flat = X_img.reshape(X_img.shape[0], -1)

        # Concatenar
        X_combined = np.concatenate([X_img_flat, X_tab], axis=1)

        X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in range(self.epochs):
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, X_img, X_tab, y):
        X_img_flat = X_img.reshape(X_img.shape[0], -1)
        X_combined = np.concatenate([X_img_flat, X_tab], axis=1)

        X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy().flatten()

        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)

        return {
            "roc_auc": roc_auc_score(y, probs),
            "accuracy": accuracy_score(y, preds)
        }