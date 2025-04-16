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

    def _euclidean_distance_squared(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Oblicza kwadrat odległości euklidesowej."""
        # Suma kwadratów różnic wzdłuż wymiaru embeddingu
        return torch.sum(torch.pow(x1 - x2, 2), dim=1)

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
    # Parametry
    import time
    batch_size = 128 # Większy batch dla potencjalnych różnic wydajności
    embedding_dim = 256
    margin_value = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Użyj GPU jeśli dostępne

    print(f"Używane urządzenie: {device}")
    print(f"Rozmiar batcha: {batch_size}, Wymiar embeddingu: {embedding_dim}, Margines: {margin_value}")

    # Losowe dane wejściowe (bardziej realistyczne niż małe, ręczne przykłady)
    anchor_batch = torch.randn(batch_size, embedding_dim, device=device, requires_grad=True)
    # Zróbmy kopię, aby gradienty się nie mieszały między testami
    anchor_batch_copy = anchor_batch.detach().clone().requires_grad_(True)

    positive_batch = torch.randn(batch_size, embedding_dim, device=device, requires_grad=True)
    positive_batch_copy = positive_batch.detach().clone().requires_grad_(True)

    negative_batch = torch.randn(batch_size, embedding_dim, device=device, requires_grad=True)
    negative_batch_copy = negative_batch.detach().clone().requires_grad_(True)


    # --- Test implementacji własnej (używając standardowej odległości) ---
    custom_loss_fn = TripletLoss(margin=margin_value, use_distance_squared=False).to(device)

    # Pomiar czasu dla forward pass
    start_time = time.time()
    loss_custom = custom_loss_fn(anchor_batch, positive_batch, negative_batch)
    custom_forward_time = time.time() - start_time

    # Pomiar czasu dla backward pass
    start_time = time.time()
    loss_custom.backward()
    custom_backward_time = time.time() - start_time
    custom_anchor_grad_norm = anchor_batch.grad.norm().item() # Sprawdźmy normę gradientu

    print(f"\n--- Własna implementacja (odległość L2) ---")
    print(f"Obliczona strata: {loss_custom.item():.6f}")
    print(f"Czas przejścia w przód (forward): {custom_forward_time:.6f} s")
    print(f"Czas przejścia w tył (backward): {custom_backward_time:.6f} s")
    print(f"Norma gradientu kotwicy: {custom_anchor_grad_norm:.6f}")


    # --- Test implementacji wbudowanej ---
    # Używamy tych samych parametrów: margin, p=2 (L2/Euklidesowa), redukcja 'mean'
    builtin_loss_fn = nn.TripletMarginLoss(margin=margin_value, p=2, reduction='mean').to(device)

    # Używamy kopii danych wejściowych
    start_time = time.time()
    loss_builtin = builtin_loss_fn(anchor_batch_copy, positive_batch_copy, negative_batch_copy)
    builtin_forward_time = time.time() - start_time

    # Pomiar czasu dla backward pass
    start_time = time.time()
    loss_builtin.backward()
    builtin_backward_time = time.time() - start_time
    builtin_anchor_grad_norm = anchor_batch_copy.grad.norm().item() # Sprawdźmy normę gradientu

    print(f"\n--- Wbudowana implementacja (nn.TripletMarginLoss, p=2) ---")
    print(f"Obliczona strata: {loss_builtin.item():.6f}")
    print(f"Czas przejścia w przód (forward): {builtin_forward_time:.6f} s")
    print(f"Czas przejścia w tył (backward): {builtin_backward_time:.6f} s")
    print(f"Norma gradientu kotwicy: {builtin_anchor_grad_norm:.6f}")


    # --- Porównanie wyników ---
    print("\n--- Porównanie ---")
    tolerance = 1e-5 # Tolerancja dla porównania floatów
    print(f"Czy wartości strat są zbliżone (tolerancja={tolerance})? ", end="")
    print(torch.isclose(loss_custom.cpu().detach(), loss_builtin.cpu().detach(), atol=tolerance).item())

    print(f"Czy normy gradientów kotwic są zbliżone (tolerancja={tolerance})? ", end="")
    print(abs(custom_anchor_grad_norm - builtin_anchor_grad_norm) < tolerance * max(custom_anchor_grad_norm, builtin_anchor_grad_norm)) # Porównanie względne

    print(f"\nRóżnica w czasie 'forward': {abs(custom_forward_time - builtin_forward_time):.6f} s")
    print(f"Różnica w czasie 'backward': {abs(custom_backward_time - builtin_backward_time):.6f} s")
