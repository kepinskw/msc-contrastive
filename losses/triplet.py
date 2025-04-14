# losses/triplet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Implementacja funkcji straty Triplet Margin Loss (FaceNet, Weinberger et al.).
    Używana do uczenia embeddingów tak, aby odległość między kotwicą (anchor)
    a przykładem pozytywnym (positive) była mniejsza niż odległość między
    kotwicą a przykładem negatywnym (negative) o co najmniej `margin`.

    L(a, p, n) = max(0, D(a, p)^2 - D(a, n)^2 + margin)
    lub
    L(a, p, n) = max(0, D(a, p) - D(a, n) + margin)

    Implementujemy wersję z kwadratami odległości. D to odległość euklidesowa.
    """
    def __init__(self, margin: float = 0.2, use_distance_squared: bool = True):
        """
        Args:
            margin (float): Margines separacji między parami (a, p) i (a, n).
            use_distance_squared (bool): Czy używać kwadratu odległości euklidesowej
                                         (True, jak często w implementacjach) czy
                                         samej odległości (False).
        """
        super().__init__()
        if margin < 0:
            raise ValueError("Margines musi być nieujemny.")
        self.margin = margin
        self.use_distance_squared = use_distance_squared
        # Uwaga: PyTorch ma wbudowaną `nn.TripletMarginLoss`, która może być alternatywą.
        # self.torch_triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')

    def _euclidean_distance_squared(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Oblicza kwadrat odległości euklidesowej."""
        # Suma kwadratów różnic wzdłuż wymiaru embeddingu
        return torch.sum(torch.pow(x1 - x2, 2), dim=-1)

    def _euclidean_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Oblicza odległość euklidesową."""
        return torch.sqrt(self._euclidean_distance_squared(x1, x2) + 1e-6) # Dodaj epsilon dla stabilności

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Oblicza stratę Triplet Loss dla batcha tripletów.

        Args:
            anchor (torch.Tensor): Batch embeddingów kotwic ([batch_size, embedding_dim]).
            positive (torch.Tensor): Batch embeddingów przykładów pozytywnych ([batch_size, embedding_dim]).
            negative (torch.Tensor): Batch embeddingów przykładów negatywnych ([batch_size, embedding_dim]).

        Returns:
            torch.Tensor: Średnia strata Triplet Loss dla batcha (skalar).
        """
        if self.use_distance_squared:
            distance_positive = self._euclidean_distance_squared(anchor, positive)
            distance_negative = self._euclidean_distance_squared(anchor, negative)
            loss = distance_positive - distance_negative + self.margin
        else:
            distance_positive = self._euclidean_distance(anchor, positive)
            distance_negative = self._euclidean_distance(anchor, negative)
            loss = distance_positive - distance_negative + self.margin

        # Zastosuj max(0, loss) używając clamp
        loss = torch.clamp(loss, min=0.0)
        mean_loss = loss.mean()

        return mean_loss


# Przykład użycia (można umieścić w bloku if __name__ == "__main__":)
if __name__ == '__main__':
    loss_fn = TripletLoss(margin=0.5, use_distance_squared=True)
    # Przykładowe embeddingi
    anchor_emb = torch.randn(4, 128, requires_grad=True)
    positive_emb = anchor_emb + torch.randn(4, 128) * 0.1 # Blisko anchor
    negative_emb = torch.randn(4, 128)                 # Losowo, dalej od anchor

    loss_value = loss_fn(anchor_emb, positive_emb, negative_emb)
    print(f"Triplet Loss: {loss_value.item()}")
    # Sprawdzenie gradientów
    # loss_value.backward()
    # print(anchor_emb.grad is not None)