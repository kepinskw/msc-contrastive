# methods/triplet_net.py
import torch
import torch.nn as nn
# from models.resnet_base import get_base_encoder # Przykład importu enkodera

class TripletNet(nn.Module):
    """
    Sieć dla Triplet Loss. Zazwyczaj używa jednego enkodera bazowego.
    Architektonicznie często identyczna z SiameseNet.
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
        print(f"Uwaga: Używam placeholdera dla base_encoder w TripletNet!")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przetwarza jedno wejście (kotwicę, pozytyw lub negatyw) przez enkoder.
        W pętli treningowej ta metoda będzie wywołana 3 razy dla każdego tripletu.

        Args:
            x: Wejście (batch obrazów - anchor, positive lub negative).

        Returns:
            Embedding dla wejścia x.
        """
        # TODO: Przepuść x przez enkoder bazowy i zwróć embedding
        embedding = self.base_encoder(x) # Placeholder
        return embedding