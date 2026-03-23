class CNNMultimodal(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        # CNN (imagen)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcular tamaño dinámico después
        self.tabular = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 13 * 13 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, tab):
        x_img = self.cnn(img)
        x_tab = self.tabular(tab)

        x = torch.cat([x_img, x_tab], dim=1)
        return self.fc(x)