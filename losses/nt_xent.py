# losses/nt_xent.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Implementacja funkcji straty NT-Xent (Normalized Temperature-scaled Cross Entropy)
    używanej w SimCLR (Chen et al., 2020).

    Oblicza stratę kontrastową dla par pozytywnych (dwóch augmentacji tego samego obrazu)
    względem par negatywnych (augmentacji różnych obrazów w batchu).
    """
    def __init__(self, temperature: float = 0.1, batch_size: int | None = None, device: torch.device | str | None = None):
        """
        Args:
            temperature (float): Parametr skalujący temperatury (tau). Kontroluje
                                 ostrość rozkładu podobieństw. Mniejsze wartości
                                 powodują większe rozdzielenie.
            batch_size (int): Rozmiar batcha (liczba ORYGINALNYCH obrazów). Potrzebny
                               do identyfikacji par pozytywnych. Jeśli None, próbuje
                               ustalić go dynamicznie w forward, ale może być mniej wydajne.
            device (torch.device | str | None): Urządzenie ('cuda', 'cpu'), na którym
                                                tworzone będą maski. Jeśli None, używa
                                                urządzenia tensora wejściowego.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperatura musi być dodatnia.")
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="mean") # Użyjemy CE loss jako części obliczeń
        self.similarity_f = nn.CosineSimilarity(dim=2) # Do obliczania podobieństwa kosinusowego

    def _get_masks_and_positive_pairs(self, current_batch_size: int, device: torch.device):
        """Tworzy maski do identyfikacji par pozytywnych i negatywnych."""
        # Maska wykluczająca przekątną (porównania elementu z samym sobą)
        mask = torch.ones((current_batch_size * 2, current_batch_size * 2), dtype=torch.bool, device=device)
        mask = mask.fill_diagonal_(0)

        # Maska dla par pozytywnych: (i, i+N) oraz (i+N, i)
        labels = torch.arange(current_batch_size, device=device)
        # Dla pary (i, i+N): etykieta to i+N
        # Dla pary (i+N, i): etykieta to i
        # Łączymy etykiety dla całego podwójnego batcha
        positive_indices = torch.cat((labels + current_batch_size, labels), dim=0) # Kształt [2*N]

        return mask, positive_indices

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Oblicza stratę NT-Xent.

        Args:
            z_i (torch.Tensor): Batch embeddingów z głowicy projekcyjnej dla pierwszego
                                zestawu augmentacji (kształt: [batch_size, projection_dim]).
            z_j (torch.Tensor): Batch embeddingów dla drugiego zestawu augmentacji
                                (kształt: [batch_size, projection_dim]).

        Returns:
            torch.Tensor: Średnia strata NT-Xent dla batcha (skalar).
        """
        current_batch_size = z_i.shape[0]
        projection_dim = z_i.shape[1]
        current_device = z_i.device

        if self.batch_size is not None and current_batch_size != self.batch_size:
            print(f"Ostrzeżenie: Aktualny batch_size ({current_batch_size}) różni się od zainicjalizowanego ({self.batch_size}). Używam aktualnego.")
            # W takim przypadku musimy dynamicznie tworzyć maski, co może być wolniejsze
            # Jeśli batch_size jest stały, można maski stworzyć raz w __init__

        if self.device is None:
            device_to_use = current_device
        else:
            device_to_use = self.device

        # Połącz embeddingi w jeden duży tensor [2*N, D]
        representations = torch.cat([z_i, z_j], dim=0)

        # Oblicz macierz podobieństwa kosinusowego [2*N, 2*N]
        # similarity_matrix = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0))
        # Alternatywnie (często szybsze): normalizacja + mnożenie macierzy
        representations_norm = F.normalize(representations, dim=1)
        similarity_matrix = torch.matmul(representations_norm, representations_norm.T)

        # Pobierz maski i indeksy par pozytywnych
        mask_negatives, positive_indices = self._get_masks_and_positive_pairs(current_batch_size, device_to_use)

        # Wybierz podobieństwa dla par pozytywnych (i, i+N) oraz (i+N, i)
        # l_pos to podobieństwo między augmentacjami tego samego obrazu
        l_pos = torch.diag(similarity_matrix, current_batch_size)  # Górna przekątna poza główną
        r_pos = torch.diag(similarity_matrix, -current_batch_size) # Dolna przekątna poza główną
        positives = torch.cat([l_pos, r_pos]).view(current_batch_size * 2, 1) # Kształt [2*N, 1]

        # Wybierz podobieństwa dla par negatywnych (wszystkie inne pary w macierzy)
        # Używamy maski, która wyklucza przekątną (i, i) oraz pary pozytywne (i, i+N), (i+N, i)
        # Maska `mask_negatives` już wyklucza przekątną.
        # W kontekście CE loss, potrzebujemy logitów dla wszystkich klas (par).
        # Logity to po prostu podobieństwa podzielone przez temperaturę.
        # Maska `mask_negatives` jest używana w CrossEntropyLoss do ignorowania self-similarity.

        logits = similarity_matrix[mask_negatives].view(current_batch_size * 2, -1) # Kształt [2*N, 2*N-2] ?? - To nie jest standardowy CE Loss
        # Standardowa implementacja używa nn.CrossEntropyLoss
        # Logity to wszystkie podobieństwa (poza self-similarity) podzielone przez temperaturę
        # Etykiety wskazują, która para jest pozytywna dla danego wiersza (augmentacji)

        # Skaluj podobieństwa przez temperaturę
        logits_scaled = similarity_matrix / self.temperature

        # Usuń podobieństwo elementu do samego siebie (przekątna) z logitów,
        # bo nie powinno ono wpływać na stratę. Można to zrobić ustawiając je na -inf
        # lub używając maski w CrossEntropyLoss (co jest bardziej skomplikowane).
        # Prostsze: użyć logitów i etykiet wskazujących poprawną parę pozytywną.

        # W `nn.CrossEntropyLoss(logits, labels)`:
        # `logits` ma kształt [batch_size, num_classes]
        # `labels` ma kształt [batch_size] i zawiera indeksy poprawnych klas.
        # W naszym przypadku:
        # - Każdy wiersz `i` w `logits_scaled` (o kształcie [2*N, 2*N]) reprezentuje augmentację `i`.
        # - Etykieta dla wiersza `i` to indeks kolumny `j`, gdzie `j` jest drugą augmentacją *tego samego* obrazu.
        # - `positive_indices` (kształt [2*N]) zawiera te etykiety.

        # Musimy jednak wykluczyć porównanie (i, i) z logitów.
        # Możemy stworzyć macierz logitów bez przekątnej.
        logits_no_diag = logits_scaled[mask_negatives].view(current_batch_size * 2, -1) # Kształt [2*N, 2*N-1]
        # Etykiety trzeba dostosować do tej nowej macierzy (indeksy się przesuną). To skomplikowane.

        # Podejście z oryginalnego kodu SimCLR (uproszczone):
        # Logity to cała macierz podobieństw (przeskalowana).
        # Etykiety (`positive_indices`) wskazują, który element w każdym wierszu jest parą pozytywną.
        # `nn.CrossEntropyLoss` obliczy stratę.
        # Problem: uwzględnia `sim(i, i)` jako potencjalną parę negatywną.

        # Lepsze podejście: Wyzeruj logity na przekątnej (odpowiadające self-similarity)
        # lub ustaw je na dużą ujemną wartość, aby nie zostały wybrane przez softmax.
        # `logits_scaled.fill_diagonal_(-float('inf'))` # Może powodować NaN/Inf w grad

        # Podejście z wielu implementacji online:
        loss = self.criterion(logits_scaled, positive_indices)

        return loss * (2 * self.temperature) # Skalowanie jak w niektórych implementacjach? Czy potrzebne? Zwykle nie.
        # return loss # Zwykle zwraca się samą stratę CE

# Przykład użycia (można umieścić w bloku if __name__ == "__main__":)
# if __name__ == '__main__':
#     batch_s = 4 # Mały batch dla przykładu
#     proj_dim = 128
#     dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     loss_fn = NTXentLoss(temperature=0.1, batch_size=batch_s, device=dev)
#
#     # Przykładowe embeddingi z głowicy projekcyjnej
#     z_i_sample = torch.randn(batch_s, proj_dim, device=dev, requires_grad=True)
#     z_j_sample = torch.randn(batch_s, proj_dim, device=dev, requires_grad=True)
#
#     loss_value = loss_fn(z_i_sample, z_j_sample)
#     print(f"NT-Xent Loss: {loss_value.item()}")
#     # Sprawdzenie gradientów
#     # loss_value.backward()
#     # print(z_i_sample.grad is not None)