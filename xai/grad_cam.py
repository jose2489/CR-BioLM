import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class MultimodalGradCAM:
    def __init__(self, model, target_layer_index=4):
        """
        Inicializa Grad-CAM conectando "ganchos" (hooks) a la red neuronal.
        En nuestra MultimodalNet, cnn_branch[4] es la última capa Conv2d (de 32 a 64 canales),
        que contiene la información espacial de más alto nivel antes de aplanarse.
        """
        self.model = model
        self.model.eval()
        self.target_layer = self.model.cnn_branch[target_layer_index]
        
        self.gradients = None
        self.activations = None

        # Conectar los ganchos para atrapar los datos mientras fluyen por la red
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, img_tensor, tab_tensor, target_class=1):
        """Genera el mapa de calor espacial para una muestra específica."""
        # Asegurarnos de que los tensores requieran gradientes para hacer el rastreo hacia atrás
        img_tensor = img_tensor.clone().detach().requires_grad_(True)
        tab_tensor = tab_tensor.clone().detach().requires_grad_(True)
        
        # 1. Paso hacia adelante (Forward)
        self.model.zero_grad()
        output = self.model(img_tensor, tab_tensor)
        
        # 2. Elegimos la clase objetivo (Presencia = 1) y hacemos el paso hacia atrás (Backward)
        score = output[0, target_class]
        score.backward()
        
        # 3. Calculamos los pesos (Promedio global de los gradientes por canal)
        # activations shape: (1, 64, 3, 3), gradients shape: (1, 64, 3, 3)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 4. Multiplicamos cada canal de activación por su peso de importancia
        activations = self.activations.detach()[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        # 5. Promediamos los canales y aplicamos ReLU (solo nos importa lo que suma positivamente a la clase)
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)
        
        # 6. Normalizamos el mapa de calor entre 0 y 1
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
            
        # 7. Redimensionamos el mapa de calor (ej. de 3x3) al tamaño original de la imagen (15x15)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0) # Formato para interpolate
        heatmap = F.interpolate(heatmap, size=(img_tensor.size(2), img_tensor.size(3)), mode='bilinear', align_corners=False)
        
        return heatmap.squeeze().cpu().numpy()

    def plot_cam(self, original_img_channel, heatmap, out_dir, filename="cnn_grad_cam.png"):
        """Superpone el mapa de calor sobre una variable climática original (ej. Temperatura)."""
        plt.figure(figsize=(8, 6))
        
        # Mostramos la variable original de fondo en escala de grises
        plt.imshow(original_img_channel, cmap='gray', alpha=0.6)
        
        # Superponemos el mapa de calor con colores cálidos
        im = plt.imshow(heatmap, cmap='jet', alpha=0.5)
        
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Importancia Espacial (Grad-CAM)")
        plt.title("Grad-CAM: Áreas que activaron la predicción de Presencia")
        plt.axis('off')
        
        out_path = os.path.join(out_dir, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Mapa de calor Grad-CAM guardado en: {out_path}")