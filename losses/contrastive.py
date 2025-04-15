# losses/contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Implementacja funkcji straty Contrastive Loss (Hadsell et al., 2006).
    Używana do uczenia embeddingów przez przyciąganie par podobnych
    i odpychanie par różnych o określony margines.

    L(x1, x2, Y) = (1-Y) * D^2 + Y * max(0, margin - D)^2
    gdzie D = ||emb(x1) - emb(x2)||_2
    Y = 0 dla pary podobnej, Y = 1 dla pary różnej.
    """
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin (float): Margines definiujący, jak daleko mają być odpychane
                            różne pary.
        """
        super().__init__()
        if margin < 0:
            raise ValueError("Margines musi być nieujemny.")
        self.margin = margin
        # Można użyć F.pairwise_distance dla uproszczenia, ale implementujemy ręcznie dla jasności
        # self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Oblicza stratę Contrastive Loss dla batcha par embeddingów.

        Args:
            embedding1 (torch.Tensor): Batch embeddingów dla pierwszych elementów par
                                       (kształt: [batch_size, embedding_dim]).
            embedding2 (torch.Tensor): Batch embeddingów dla drugich elementów par
                                       (kształt: [batch_size, embedding_dim]).
            label (torch.Tensor): Etykiety par (0 dla podobnych, 1 dla różnych)
                                  (kształt: [batch_size]).

        Returns:
            torch.Tensor: Średnia strata Contrastive Loss dla batcha (skalar).
        """
        # Oblicz odległość euklidesową między parami embeddingów
        # D = ||emb1 - emb2||_2
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, p=2, keepdim=False)

        # Oblicz stratę zgodnie ze wzorem
        # Strata dla par podobnych (label == 0): distance^2
        loss_similar = euclidean_distance.pow(2)

        # Strata dla par różnych (label == 1): max(0, margin - distance)^2
        loss_dissimilar = torch.clamp(self.margin - euclidean_distance, min=0.0).pow(2)

        label_float = label.float()
        loss = (1.0 - label_float) * loss_similar + label_float * loss_dissimilar

        # Zwróć średnią stratę dla całego batcha
        mean_loss = torch.mean(loss)
        return mean_loss

# Przykład użycia (można umieścić w bloku if __name__ == "__main__":)
# if __name__ == '__main__':
#     loss_fn = ContrastiveLoss(margin=1.0)
#     emb1 = torch.randn(4, 128, requires_grad=True)
#     emb2 = torch.randn(4, 128, requires_grad=True)
#     # Przykładowe etykiety: [podobna, różna, podobna, różna]
#     labels = torch.tensor([0, 1, 0, 1])
#
#     loss_value = loss_fn(emb1, emb2, labels)
#     print(f"Contrastive Loss: {loss_value.item()}")
#     # Sprawdzenie gradientów
#     # loss_value.backward()
#     # print(emb1.grad is not None)