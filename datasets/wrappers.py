import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict
import numpy as np
from PIL import Image

class TripletDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform=None):
        """
        Args:
            base_dataset (Dataset): Instancja bazowego datasetu PyTorch. Oczekuje się,
                                   że ma atrybut przechowujący etykiety (np. .targets, .labels)
                                   oraz metodę __getitem__ zwracającą surowy obraz
                                   lub dane, z których można uzyskać obraz i etykietę.
                                   Najlepiej, jeśli base_dataset jest przekazany *bez*
                                   własnych transformacji obrazu (transformacje stosuje wrapper).
            transform (callable, optional): Transformacja (augmentacja) do zastosowania
                                           niezależnie na obrazach anchor, positive i negative.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

        # Sprawdź, czy base_dataset ma etykiety i dane
        self.labels = self._get_labels_from_dataset()
        self.data = self._get_data_from_dataset() # Może być None, jeśli dostęp tylko przez __getitem__

        if self.labels is None:
            raise ValueError("Nie można automatycznie pobrać etykiet z base_dataset. Sprawdź atrybuty .targets lub .labels.")

        # Organizujemy indeksy danych według klas
        print("Organizowanie indeksów według klas dla TripletDatasetWrapper...")
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            # Konwertuj etykietę na typ podstawowy, jeśli to tensor/numpy scalar
            label_item = label.item() if isinstance(label, (torch.Tensor, np.generic)) else label
            self.label_to_indices[label_item].append(idx)

        self.unique_labels = sorted(list(self.label_to_indices.keys()))

        # Sprawdzenie, czy są klasy z tylko jedną próbką
        for label, indices in self.label_to_indices.items():
            if len(indices) < 2:
                print(f"Ostrzeżenie: Klasa {label} w base_dataset ma tylko {len(indices)} próbkę/próbki. "
                      "Wybór 'positive' może być niemożliwy lub powtarzalny.")
        print("Organizacja indeksów zakończona.")


    def _get_labels_from_dataset(self):
        """Próbuje pobrać etykiety z różnych potencjalnych atrybutów."""
        if hasattr(self.base_dataset, 'targets'):
            return np.array(self.base_dataset.targets) # np. CIFAR10, MNIST
        elif hasattr(self.base_dataset, 'labels'):
            return np.array(self.base_dataset.labels) # np. SVHN
        elif hasattr(self.base_dataset, '_labels'): # Czasem prywatne atrybuty
             return np.array(self.base_dataset._labels)
        else:
            # Ostateczność: spróbuj iterować (może być BARDZO wolne dla dużych datasetów)
            print("Ostrzeżenie: Próbuję pobrać etykiety iterując po base_dataset. To może być wolne.")
            try:
                 return np.array([self.base_dataset[i][1] for i in range(len(self.base_dataset))])
            except Exception as e:
                 print(f"Błąd podczas iteracyjnego pobierania etykiet: {e}")
                 return None

    def _get_data_from_dataset(self):
        """Próbuje pobrać dane obrazów, jeśli są dostępne jako atrybut."""
        if hasattr(self.base_dataset, 'data'):
            return self.base_dataset.data # np. CIFAR10, SVHN
        # Można dodać inne sprawdzenia, np. dla ImageFolder (base_dataset.samples)
        return None # Jeśli dane dostępne tylko przez __getitem__

    def _get_image_by_index(self, index: int):
        """Pobiera surowy obraz (PIL lub ndarray) dla danego indeksu."""
        if self.data is not None:
            # Bezpośredni dostęp, jeśli mamy atrybut .data (np. CIFAR10, SVHN zwraca ndarray)
            img_data = self.data[index]
            # Upewnij się, że zwracamy PIL Image, jeśli to potrzebne dla transformacji
            if isinstance(img_data, np.ndarray):
                 # CIFAR/SVHN zwraca (H, W, C) lub (C, H, W) - dostosuj transpozycję
                 if img_data.shape[-1] == 3: # Zakładamy H, W, C
                     return Image.fromarray(img_data)
                 elif img_data.shape[0] == 3: # Zakładamy C, H, W
                     return Image.fromarray(np.transpose(img_data, (1, 2, 0)))
                 else:
                     # Spróbuj domyślnie
                     try: return Image.fromarray(img_data)
                     except Exception as e: print(f"Nie można przekonwertować danych z indeksu {index} na PIL Image: {e}") ; return None
            elif isinstance(img_data, Image.Image):
                 return img_data
            else:
                 # Spróbuj przez __getitem__ bazowego datasetu, zakładając, że zwraca (img, label)
                 try: return self.base_dataset[index][0] # Może zawierać już transformację - problematyczne
                 except Exception: return None # Nie udało się pobrać obrazu
        else:
            # Pobierz przez __getitem__ bazowego datasetu
            # UWAGA: To może zwrócić już przetransformowany obraz, jeśli bazowy dataset ma transformację!
            # Najlepiej przekazywać bazowy dataset BEZ transformacji.
            try:
                 img_data, _ = self.base_dataset[index]
                 return img_data # Zwróć cokolwiek zwróci __getitem__[0]
            except Exception as e:
                 print(f"Nie można pobrać obrazu dla indeksu {index} przez base_dataset.__getitem__: {e}")
                 return None


    def __len__(self):
        """Zwraca liczbę potencjalnych kotwic (rozmiar bazowego datasetu)."""
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Zwraca triplet (anchor_img, positive_img, negative_img)
        po zastosowaniu transformacji.
        """
        # 1. Pobierz etykietę kotwicy
        anchor_label_item = self.labels[index].item() if isinstance(self.labels[index], (torch.Tensor, np.generic)) else self.labels[index]

        # 2. Wybierz indeks pozytywny (inny niż anchor)
        possible_positive_indices = self.label_to_indices[anchor_label_item]
        # Usuń bieżący indeks z możliwych pozytywów
        positive_candidate_indices = [idx_ for idx_ in possible_positive_indices if idx_ != index]
        positive_idx = random.choice(positive_candidate_indices)

        # 3. Wybierz indeks negatywny
        possible_negative_labels = [label for label in self.unique_labels if label != anchor_label_item]
        negative_label = random.choice(possible_negative_labels)
        negative_idx = random.choice(self.label_to_indices[negative_label])

        # 4. Pobierz surowe obrazy dla anchor, positive, negative
        anchor_img_raw = self._get_image_by_index(index)
        positive_img_raw = self._get_image_by_index(positive_idx)
        negative_img_raw = self._get_image_by_index(negative_idx)

        # Sprawdzenie, czy obrazy zostały pobrane poprawnie
        if anchor_img_raw is None or positive_img_raw is None or negative_img_raw is None:
            print(f"OSTRZEŻENIE: Nie udało się pobrać wszystkich obrazów dla tripletu z anchor index {index}. Zwracam None.")
            raise RuntimeError(f"Nie udało się pobrać obrazów dla tripletu z anchor index {index}")


        # 5. Zastosuj transformacje (jeśli istnieją)
        if self.transform:
            anchor_img = self.transform(anchor_img_raw)
            positive_img = self.transform(positive_img_raw)
            negative_img = self.transform(negative_img_raw)
        else:
             raise ValueError("Transformacja (zawierająca ToTensor) jest wymagana dla TripletDatasetWrapper.")


        return anchor_img, positive_img, negative_img
    


#####
def _get_labels_from_dataset(base_dataset):
    """Próbuje pobrać etykiety z różnych potencjalnych atrybutów."""
    if hasattr(base_dataset, 'targets'): return np.array(base_dataset.targets)
    if hasattr(base_dataset, 'labels'): return np.array(base_dataset.labels)
    if hasattr(base_dataset, '_labels'): return np.array(base_dataset._labels)
    print("Ostrzeżenie: Próbuję pobrać etykiety iterując po base_dataset.")
    try: return np.array([base_dataset[i][1] for i in range(len(base_dataset))])
    except Exception as e: print(f"Błąd: {e}"); return None

def _get_data_from_dataset(base_dataset):
    """Próbuje pobrać dane obrazów, jeśli są dostępne jako atrybut."""
    if hasattr(base_dataset, 'data'): return base_dataset.data
    return None

def _get_image_by_index(base_dataset, index, data_attr):
    """Pobiera surowy obraz (PIL lub ndarray) dla danego indeksu."""
    # Zakłada, że base_dataset[index] zwraca (raw_image, label)
    # lub że raw_image jest w data_attr[index]
    if data_attr is not None:
        img_data = data_attr[index]
        if isinstance(img_data, np.ndarray):
            if img_data.shape[-1] == 3: return Image.fromarray(img_data)
            if img_data.shape[0] == 3: return Image.fromarray(np.transpose(img_data, (1, 2, 0)))
            try: return Image.fromarray(img_data)
            except Exception as e: print(f"Nie można przekonwertować numpy z indeksu {index}: {e}"); return None
        if isinstance(img_data, Image.Image): return img_data
        # Jeśli dane są innego typu, ta funkcja wymagałaby dostosowania
        print(f"Ostrzeżenie: Nieobsługiwany typ danych w .data dla indeksu {index}: {type(img_data)}")
        # Spróbuj pobrać przez __getitem__ jako fallback
        try: return base_dataset[index][0]
        except Exception: return None
    else:
        # Pobierz przez __getitem__ bazowego datasetu
        try:
            img_data, _ = base_dataset[index] # Pobierz obraz, ignoruj etykietę
            # Sprawdź, czy to PIL Image - jeśli nie, może wymagać konwersji
            if not isinstance(img_data, Image.Image):
                 # Spróbuj konwersji z numpy, jeśli to możliwe
                 if isinstance(img_data, np.ndarray):
                     return _get_image_by_index(base_dataset, index, np.expand_dims(img_data, axis=0)) # Wywołaj ponownie z ndarray
                 else:
                     print(f"Ostrzeżenie: Obraz z base_dataset[{index}] nie jest typu PIL.Image ani ndarray ({type(img_data)}). Transformacje mogą nie działać.")
            return img_data
        except Exception as e:
            print(f"Nie można pobrać obrazu dla indeksu {index} przez base_dataset.__getitem__: {e}")
            return None
        

class SiameseDataset(Dataset):
    """
    Wrapper dla bazowego datasetu PyTorch (zwracającego image, label),
    który generuje pary (image1, image2, pair_label) na potrzeby
    treningu sieci syjamskich z Contrastive Loss.
    pair_label = 0 dla pary podobnej (ta sama klasa), 1 dla pary różnej.
    """
    def __init__(self, base_dataset: Dataset, transform=None, positive_fraction: float = 0.5):
        """
        Args:
            base_dataset (Dataset): Instancja bazowego datasetu PyTorch. Oczekuje się,
                                   że ma atrybut z etykietami (np. .targets, .labels)
                                   i zwraca surowy obraz w __getitem__.
                                   Najlepiej przekazać go *bez* własnych transformacji.
            transform (callable, optional): Transformacja (augmentacja) do zastosowania
                                           niezależnie na obrazach image1 i image2.
            positive_fraction (float): Przybliżona frakcja par pozytywnych generowanych
                                      (domyślnie 0.5 = 50%).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        if not 0.0 <= positive_fraction <= 1.0:
             raise ValueError("positive_fraction musi być pomiędzy 0.0 a 1.0")
        self.positive_fraction = positive_fraction

        # Pobierz etykiety i dane
        self.labels = _get_labels_from_dataset(self.base_dataset)
        self.data = _get_data_from_dataset(self.base_dataset) # Może być None

        if self.labels is None:
            raise ValueError("Nie można automatycznie pobrać etykiet z base_dataset.")

        # Organizujemy indeksy danych według klas
        print("Organizowanie indeksów według klas dla SiameseDataset...")
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            label_item = label.item() if isinstance(label, (torch.Tensor, np.generic)) else label
            self.label_to_indices[label_item].append(idx)

        self.unique_labels = sorted(list(self.label_to_indices.keys()))

        # Sprawdzenie klas z tylko jedną próbką (ważne dla par pozytywnych)
        for label, indices in self.label_to_indices.items():
            if len(indices) < 2:
                print(f"Ostrzeżenie: Klasa {label} w base_dataset ma tylko {len(indices)} próbkę/próbki. "
                      "Wybór pary pozytywnej dla tej klasy będzie niemożliwy/powtarzalny.")
        print("Organizacja indeksów zakończona.")


    def __len__(self):
        """Zwraca rozmiar bazowego datasetu."""
        return len(self.base_dataset)

    def __getitem__(self, index1: int):
        """
        Zwraca parę (image1, image2, pair_label) po zastosowaniu transformacji.
        """
        image1_raw, image2_raw = None, None
        label1_item, index2, pair_label = None, None, None

        try:
            # --- Pobranie etykiety dla obrazu 1 (anchor) ---
            label1_item = self.labels[index1].item() if isinstance(self.labels[index1], (torch.Tensor, np.generic)) else self.labels[index1]

            # --- Decyzja o typie pary i wybór indeksu 2 ---
            should_get_positive = random.random() < self.positive_fraction

            if should_get_positive:
                # Wybierz indeks pozytywny (inny niż index1)
                possible_positive_indices = self.label_to_indices[label1_item]
                positive_candidate_indices = [idx_ for idx_ in possible_positive_indices if idx_ != index1]

                if not positive_candidate_indices:
                    index2 = index1 # Edge case: użyj tego samego indeksu
                else:
                    index2 = random.choice(positive_candidate_indices)
                pair_label = 0 # Para podobna
            else:
                # Wybierz indeks negatywny
                possible_negative_labels = [label for label in self.unique_labels if label != label1_item]
                if not possible_negative_labels:
                     # Edge case: tylko jedna klasa w datasecie
                     print(f"Ostrzeżenie: Brak innych klas dla index {index1}. Zwracam parę pozytywną.")
                     possible_positive_indices = self.label_to_indices[label1_item]
                     positive_candidate_indices = [idx_ for idx_ in possible_positive_indices if idx_ != index1]
                     if not positive_candidate_indices: index2 = index1
                     else: index2 = random.choice(positive_candidate_indices)
                     pair_label = 0 # Zwróć pozytywną zamiast negatywnej
                else:
                     negative_label = random.choice(possible_negative_labels)
                     index2 = random.choice(self.label_to_indices[negative_label])
                     pair_label = 1 # Para różna

            # --- Pobranie surowych obrazów ---
            image1_raw = _get_image_by_index(self.base_dataset, index1, self.data)
            image2_raw = _get_image_by_index(self.base_dataset, index2, self.data)

            # Sprawdzenie
            if image1_raw is None or image2_raw is None:
                missing = []
                if image1_raw is None: missing.append(f"image1 (idx={index1})")
                if image2_raw is None: missing.append(f"image2 (idx={index2})")
                raise ValueError(f"Nie udało się pobrać obrazów: {', '.join(missing)}")

        except Exception as e:
            print(f"BŁĄD KRYTYCZNY w __getitem__ dla index {index1} podczas wybierania/pobierania pary: {e}")
            raise RuntimeError(f"Błąd w __getitem__ dla index {index1}") from e


        # --- Zastosowanie Transformacji ---
        if self.transform:
            try:
                # Konwersja do PIL (jeśli potrzebne)
                if not isinstance(image1_raw, Image.Image):
                     if isinstance(image1_raw, np.ndarray): image1_raw = Image.fromarray(image1_raw)
                     else: raise TypeError(f"Nieobsługiwany typ dla image1 (idx {index1}): {type(image1_raw)}")
                if not isinstance(image2_raw, Image.Image):
                     if isinstance(image2_raw, np.ndarray): image2_raw = Image.fromarray(image2_raw)
                     else: raise TypeError(f"Nieobsługiwany typ dla image2 (idx {index2}): {type(image2_raw)}")

                # Aplikacja transformacji
                image1 = self.transform(image1_raw)
                image2 = self.transform(image2_raw)
            except Exception as e:
                print(f"BŁĄD KRYTYCZNY podczas transformacji obrazów dla pary z index1={index1}. Błąd: {e}")
                raise RuntimeError(f"Błąd transformacji dla index {index1}") from e
        else:
            raise ValueError("Transformacja (zawierająca ToTensor) jest wymagana dla SiameseDataset.")

        # Zwróć przetransformowaną parę i etykietę
        return image1, image2, pair_label


class SimCLRDataset(Dataset):
    """
    Wrapper Dataset dla metody SimCLR.

    Przyjmuje bazowy dataset oraz funkcję transformacji. Oczekuje się,
    że funkcja transformacji przyjmie jeden obraz (np. PIL Image)
    i zwróci *dwie* jego augmentowane wersje (widoki).

    Args:
        base_dataset (Dataset): Podstawowy dataset (np. instancja RefactoredCIFAR10),
                                 który zwraca krotki (obraz, etykieta). Zakłada się,
                                 że obraz jest w formacie akceptowanym przez transformację
                                 (np. PIL Image).
        transform (callable): Funkcja transformacji (np. instancja SimCLRTransform),
                              która przyjmuje obraz i zwraca krotkę dwóch
                              augmentowanych widoków (view1, view2).
    """
    def __init__(self, base_dataset: Dataset, transform: callable):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

        if not callable(self.transform):
            raise TypeError("Przekazany 'transform' musi być obiektem wywoływalnym (callable).")

        # Można dodać walidację sprawdzającą, czy transformacja zwraca 2 elementy,
        # ale często polega się na poprawnym skonfigurowaniu przez użytkownika.
        # Przykład (może być kruchy):
        # try:
        #     img_test, _ = base_dataset[0] # Pobierz przykładowy obraz
        #     if not isinstance(img_test, Image.Image): # Jeśli nie PIL, dostosuj
        #          img_test = Image.fromarray(img_test) # Przykładowa konwersja z numpy
        #     view1, view2 = self.transform(img_test)
        # except Exception as e:
        #     print(f"Ostrzeżenie: Nie udało się zweryfikować wyjścia transformacji: {e}")
        #     pass


    def __getitem__(self, index):
        """
        Pobiera element z bazowego datasetu, stosuje transformację SimCLR
        i zwraca dwa augmentowane widoki oraz oryginalną etykietę.

        Args:
            index (int): Indeks elementu w bazowym datasecie.

        Returns:
            tuple: (view1, view2, target)
                   - view1: Pierwszy augmentowany widok obrazu.
                   - view2: Drugi augmentowany widok obrazu.
                   - target: Oryginalna etykieta z bazowego datasetu.
        """
        # Pobierz obraz i etykietę z bazowego datasetu.
        # Zakładamy, że base_dataset[index] zwraca (dane_obrazu, etykieta)
        # i dane_obrazu są w formacie oczekiwanym przez self.transform (np. PIL Image).
        # Jeśli base_dataset to np. RefactoredCIFAR10(transform=None), to tak będzie.
        img, target = self.base_dataset[index]

        # Zastosuj transformację SimCLR (która generuje 2 widoki)
        # `self.transform` musi być obiektem np. klasy SimCLRTransform
        view1, view2 = self.transform(img)

        # Zwróć oba widoki i etykietę (etykieta może być przydatna później lub dla spójności)
        return view1, view2, target

    def __len__(self):
        """
        Zwraca całkowitą liczbę próbek w bazowym datasecie.
        """
        return len(self.base_dataset)