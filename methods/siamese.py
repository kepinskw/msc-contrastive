# methods/siamese.py
import torch
import torch.nn as nn
# from models.resnet_base import get_base_encoder # Przykład importu enkodera

class SiameseNet(nn.Module):
    """
    Sieć Syjamska. Używa jednego enkodera bazowego do przetwarzania par wejść.
    """
    def __init__(self, base_encoder_class):
        """
        Args:
            base_encoder_class: Klasa enkodera bazowego.
        """
        super().__init__()

        # TODO: Zainicjalizuj JEDEN enkoder bazowy - jego wagi będą współdzielone
        # self.base_encoder = base_encoder_class(pretrained=False) # Przykład
        base_output_dim = 2048 # Placeholder
        self.base_encoder = nn.Sequential( # Placeholder - zastąp prawdziwym enkoderem
             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
             # ...
             nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, base_output_dim)
        )
        print(f"Uwaga: Używam placeholdera dla base_encoder w SiameseNet!")

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Przetwarza jedno wejście przez enkoder bazowy."""
        # TODO: Przepuść x przez enkoder bazowy i zwróć embedding
        return self.base_encoder(x) # Placeholder

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Przetwarza parę wejść (x1, x2) przez współdzielony enkoder.

        Args:
            x1: Pierwszy element pary (batch obrazów).
            x2: Drugi element pary (batch obrazów).

        Returns:
            Tuple: (embedding1, embedding2) - embeddingi dla x1 i x2.
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2