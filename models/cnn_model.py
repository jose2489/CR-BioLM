import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np

# ==========================================
# 1. LA ARQUITECTURA DE LA RED NEURONAL
# ==========================================
class MultimodalNet(nn.Module):
    def __init__(self, num_img_channels=19, num_tab_features=5):
        super(MultimodalNet, self).__init__()
        
        # --- RAMA A: El Ojo Espacial (CNN para el Clima 15x15) ---
        self.cnn_branch = nn.Sequential(
            # Capa 1: Entran 19 variables, salen 32 filtros matemáticos
            nn.Conv2d(in_channels=num_img_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduce de 15x15 a 7x7
            
            # Capa 2: Extracción de patrones más complejos
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce de 7x7 a 3x3
        )
        
        # Calculamos el tamaño del vector aplanado de la imagen (64 canales * 3 alto * 3 ancho = 576)
        cnn_flatten_size = 64 * 3 * 3
        
        # --- RAMA B: El Ojo Categórico (Tabular para SINAC) ---
        self.tab_branch = nn.Sequential(
            nn.Linear(num_tab_features, 16),
            nn.ReLU()
        )
        
        # --- LA FUSIÓN TARDÍA (Late Fusion) ---
        # Sumamos el tamaño de la imagen aplanada (576) + el vector tabular (16) = 592
        fusion_size = cnn_flatten_size + 16
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Previene el sobreajuste (overfitting)
            nn.Linear(64, 2) # Salida: 2 clases (Ausencia=0, Presencia=1)
        )

    def forward(self, img_x, tab_x):
        # 1. Procesar la imagen
        img_out = self.cnn_branch(img_x)
        img_out = img_out.view(img_out.size(0), -1) # Aplanar la imagen (Flatten)
        
        # 2. Procesar la tabla
        tab_out = self.tab_branch(tab_x)
        
        # 3. Concatenar (Unir ambos vectores)
        fused = torch.cat((img_out, tab_out), dim=1)
        
        # 4. Decisión Final
        out = self.fusion_classifier(fused)
        return out


# ==========================================
# 2. EL CONTROLADOR DEL MODELO
# ==========================================
class CNNSDM:
    """Envoltorio para entrenar y evaluar la red neuronal igual que el Random Forest."""
    def __init__(self, epochs=30, batch_size=32, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        # Usaremos GPU si está disponible, si no, CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Modelo CNN inicializado. Dispositivo de cálculo: {self.device}")

    def train(self, X_img_train, X_tab_train, y_train):
        """Entrena la red multimodal."""
        print("[INFO] Entrenando Red Neuronal Convolucional (Deep Learning)...")
        
        num_img_channels = X_img_train.shape[1]
        num_tab_features = X_tab_train.shape[1]
        
        self.model = MultimodalNet(num_img_channels, num_tab_features).to(self.device)
        
        # Convertir datos a tensores de PyTorch
        img_tensor = torch.tensor(X_img_train, dtype=torch.float32)
        tab_tensor = torch.tensor(X_tab_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # Crear el cargador de datos (DataLoader)
        dataset = TensorDataset(img_tensor, tab_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Función de pérdida y optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Bucle de entrenamiento
        self.model.train()
        for epoch in range(self.epochs):
            loss_acumulada = 0.0
            for batch_img, batch_tab, batch_y in dataloader:
                batch_img, batch_tab, batch_y = batch_img.to(self.device), batch_tab.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_img, batch_tab)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                loss_acumulada += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"       -> Época [{epoch+1}/{self.epochs}], Pérdida (Loss): {loss_acumulada/len(dataloader):.4f}")
                
        print("[INFO] Entrenamiento de CNN completado exitosamente.")

    def evaluate(self, X_img_test, X_tab_test, y_test):
        """Evalúa la red y retorna el diccionario de métricas."""
        print("[INFO] Evaluando rendimiento de la CNN...")
        self.model.eval()
        
        img_tensor = torch.tensor(X_img_test, dtype=torch.float32).to(self.device)
        tab_tensor = torch.tensor(X_tab_test, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor, tab_tensor)
            # Aplicar Softmax para obtener probabilidades
            probabilidades = torch.softmax(outputs, dim=1).cpu().numpy()
            predicciones = np.argmax(probabilidades, axis=1)
            
            # Probabilidad de la clase positiva (presencia = 1)
            prob_presencia = probabilidades[:, 1]
            
        acc = accuracy_score(y_test, predicciones)
        try:
            roc_auc = roc_auc_score(y_test, prob_presencia)
        except ValueError:
            roc_auc = float('nan')
        conf_matrix = confusion_matrix(y_test, predicciones)

        print(f"[METRICAS CNN] Exactitud (Accuracy) : {acc:.4f}")
        print(f"[METRICAS CNN] Area bajo la curva (ROC-AUC) : {roc_auc:.4f}")
        print(f"[METRICAS CNN] Matriz de Confusion :\n{conf_matrix}")

        return {
            'accuracy': acc,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'probabilities': prob_presencia # Útil por si lo ocupamos luego
        }