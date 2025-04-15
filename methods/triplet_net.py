# methods/triplet_net.py
import torch
import torch.nn as nn
from models.resnet_base import get_resnet_encoder # Przykład importu enkodera

class TripletNet(nn.Module):
    """
    Sieć dla Triplet Loss. Zazwyczaj używa jednego enkodera bazowego.
    Architektonicznie często identyczna z SiameseNet.
    """
    def __init__(self, base_encoder_class=get_resnet_encoder):
        """
        Args:
            base_encoder_class: Klasa enkodera bazowego.
        """
        super().__init__()
        self.base_encoder = base_encoder_class(pretrained=False)
        self.embedding_dim = self.base_encoder.output_dim # Przykład
     


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przetwarza jedno wejście (kotwicę, pozytyw lub negatyw) przez enkoder.
        W pętli treningowej ta metoda będzie wywołana 3 razy dla każdego tripletu.

        Args:
            x: Wejście (batch obrazów - anchor, positive lub negative).

        Returns:
            Embedding dla wejścia x.
        """
        embedding = self.base_encoder(x) # Placeholder
        return embedding