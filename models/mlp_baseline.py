
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score


class MLPBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MLPTrainer:
    def __init__(self, input_dim, epochs=30, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPBaseline(input_dim).to(self.device)
        self.epochs = epochs
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X, y):
        print("[INFO] Entrenando MLP Baseline...")

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()

            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, X, y):
        self.model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy().flatten()

        preds = (probs > 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y, probs),
            "accuracy": accuracy_score(y, preds)
        }

        print(f"[MLP] ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"[MLP] Accuracy: {metrics['accuracy']:.4f}")

        return metrics