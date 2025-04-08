# models/projection_head.py

import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Głowica projekcyjna (MLP) używana w metodach takich jak SimCLR.
    Mapuje wektor cech z enkodera bazowego na niższowymiarową przestrzeń,
    w której obliczana jest strata kontrastowa.
    """
    def __init__(self, input_dim: int, hidden_dim: int | None = None, output_dim: int = 128):
        """
        Args:
            input_dim (int): Wymiar wektora cech z enkodera bazowego (np. 2048 dla ResNet50).
            hidden_dim (int | None): Wymiar warstwy ukrytej. Jeśli None, używany jest input_dim.
                                      W oryginalnym SimCLR było to równe input_dim.
            output_dim (int): Wymiar wyjściowy projekcji (np. 128 w SimCLR).
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        # Definicja warstw MLP
        # Oryginalny SimCLR: Linear -> ReLU -> Linear
        # Czasami dodaje się BatchNorm, ale w oryginale go nie było w głowicy.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        print(f"Inicjalizacja ProjectionHead: {input_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przepuszcza wektor cech przez MLP.

        Args:
            x: Wektor cech z enkodera bazowego (o wymiarze input_dim).

        Returns:
            Wektor cech po projekcji (o wymiarze output_dim).
        """
        return self.mlp(x)

# Przykład użycia (można umieścić w bloku if __name__ == "__main__":)
# if __name__ == '__main__':
#     # Przykładowy wektor cech z ResNet50 (batch=4, dim=2048)
#     dummy_features = torch.randn(4, 2048)
#
#     # Utwórz głowicę projekcyjną
#     # projection_head = ProjectionHead(input_dim=2048) # Użyje hidden_dim=2048, output_dim=128
#     projection_head = ProjectionHead(input_dim=2048, hidden_dim=512, output_dim=256)
#
#     # Przepuść cechy przez głowicę
#     projected_features = projection_head(dummy_features)
#     print(f"Kształt cech po projekcji: {projected_features.shape}") # Oczekiwano: [4, 256] w tym przykładzie