# main_eval.py

import argparse
import os
import time
import datetime
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Dodaj główny katalog projektu do ścieżki Pythona
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.cifar10 import get_cifar10_dataloader, get_cifar10_dataset
from datasets.svhn import get_svhn_dataloader, get_svhn_dataset
from datasets.celeba import get_celeba_dataloader, get_celeba_dataset
from datasets.imagenet_subset import get_imagenet_subset_dataloader, get_imagenet_subset_dataset
from models.resnet_base import get_resnet_encoder
from models.projection_head import ProjectionHead
from methods.simclr import SimCLRNet
from methods.siamese import SiameseNet
from methods.triplet_net import TripletNet
from losses.nt_xent import NTXentLoss
from losses.contrastive import ContrastiveLoss
from losses.triplet import TripletLoss


from torch.utils.tensorboard import SummaryWriter


from torchvision import transforms, datasets

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

#--- test classificator
class MLPClassifier(nn.Module):
    """
    Prosty 3-warstwowy klasyfikator MLP (Multi-Layer Perceptron).
    Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear (Logits)
    """
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int, dropout_prob: float = 0.5):
        """
        Args:
            input_dim (int): Wymiar wejściowy (wyjście enkodera bazowego).
            hidden_dim1 (int): Wymiar pierwszej warstwy ukrytej.
            hidden_dim2 (int): Wymiar drugiej warstwy ukrytej.
            output_dim (int): Wymiar wyjściowy (liczba klas).
            dropout_prob (float): Prawdopodobieństwo dropoutu.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fc3 = nn.Linear(hidden_dim2, output_dim) # Warstwa wyjściowa (logity)

        print(f"Utworzono MLPClassifier: {input_dim} -> {hidden_dim1} -> {hidden_dim2} -> {output_dim} (dropout={dropout_prob})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przepuszcza wektor cech przez MLP.

        Args:
            x: Wektor cech z enkodera bazowego (kształt: [batch_size, input_dim]).

        Returns:
            Logity dla każdej klasy (kształt: [batch_size, output_dim]).
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x) # Ostatnia warstwa zwraca logity
        return x

# ----------- Argument Parsing -----------

def parse_arguments():
    """Paruje argumenty linii poleceń dla ewaluacji (Linear Probing)."""
    parser = argparse.ArgumentParser(description='Linear Evaluation (Probing) Script')

    # Dataset Arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'svhn', 'celeba', 'imagenet_subset'],
                        help='Nazwa datasetu do ewaluacji.')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Ścieżka do katalogu z danymi.')
    parser.add_argument('--imagenet_subset_path', type=str, default='/path/to/imagenet_subset',
                        help='Ścieżka do podzbioru ImageNet (jeśli używany).')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Liczba klas w zbiorze danych ewaluacyjnych (np. 10 dla CIFAR10).')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Rozmiar obrazu (np. 32 dla CIFAR/SVHN, 128/224 dla CelebA/ImageNet).')

    # Model Arguments
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='Architektura enkodera bazowego użytego do pre-treningu.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Ścieżka do zapisanego checkpointu enkodera bazowego (.pth).')

    # Linear Evaluation Training Arguments
    parser.add_argument('--eval_epochs', type=int, default=100,
                        help='Liczba epok treningu klasyfikatora liniowego.')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='Rozmiar batcha dla treningu/ewaluacji liniowej.')
    parser.add_argument('--eval_optimizer', type=str, default='sgd', choices=['adam', 'sgd'],
                        help='Optymalizator dla klasyfikatora liniowego.')
    parser.add_argument('--eval_lr', type=float, default=0.1, # Często wyższy LR dla SGD w linear probing
                        help='Szybkość uczenia dla klasyfikatora liniowego.')
    parser.add_argument('--eval_weight_decay', type=float, default=0.0, # Zwykle brak lub mały weight decay dla głowy liniowej
                        help='Współczynnik L2 regularyzacji dla klasyfikatora liniowego.')
    parser.add_argument('--eval_scheduler', type=str, default='step', choices=['none', 'step', 'cosine'],
                        help='Typ harmonogramu uczenia dla klasyfikatora liniowego.')
    parser.add_argument('--eval_lr_decay_rate', type=float, default=0.1, help='Współczynnik zmniejszenia LR dla StepLR')
    parser.add_argument('--eval_lr_decay_epochs', type=str, default='60,80', help='Epoki do zmniejszenia LR dla StepLR (oddzielone przecinkami)')


    # System Arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Urządzenie do treningu/ewaluacji (cuda/cpu).')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Liczba wątków roboczych dla DataLoader.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Ziarno losowości dla reprodukowalności.')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Co ile kroków logować postęp treningu głowy liniowej.')

    return parser.parse_args()

# ----------- Helper Functions -----------

def set_seed(seed):
    """Ustawia ziarno losowości dla reprodukowalności."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

def load_encoder(args, device):
    print(f"--- Ładowanie Enkodera ---")
    print(f"Architektura: {args.arch}")
    # 1. Stwórz instancję architektury enkodera (bez wag z torchvision)
    encoder = get_resnet_encoder(name=args.arch, pretrained=False)
    if not hasattr(encoder, 'output_dim'):
         # Jeśli get_resnet_encoder nie dodał atrybutu, spróbujmy go ustalić (ryzykowne)
         try:
             dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
             output_shape = encoder(dummy_input).shape
             encoder.output_dim = output_shape[-1]
             print(f"Ostrzeżenie: Atrybut 'output_dim' nie znaleziony. Ustalono na {encoder.output_dim} na podstawie wyjścia.")
         except Exception:
             # Użyj domyślnych, jeśli wszystko inne zawiedzie
             encoder.output_dim = 2048 if '50' in args.arch or '101' in args.arch or '152' in args.arch else 512
             print(f"Ostrzeżenie: Nie można ustalić 'output_dim'. Ustawiono domyślnie na {encoder.output_dim}.")


    # 2. Sprawdź i załaduj plik checkpointu
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Plik checkpointu nie znaleziony: {args.checkpoint_path}")

    print(f"Ładowanie checkpointu z: {args.checkpoint_path}")
    try:
        # Załaduj cały słownik checkpointu
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu') # Załaduj najpierw na CPU

        # --- POCZĄTEK KLUCZOWEJ ZMIANY ---
        # 3. Wyodrębnij state_dict modelu z checkpointu
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("  Wyodrębniono 'model_state_dict' z checkpointu.")
            # Opcjonalnie: Wyświetl inne informacje z checkpointu
            if 'epoch' in checkpoint: print(f"  Checkpoint zapisany po epoce: {checkpoint['epoch']}")
            if 'best_loss' in checkpoint: print(f"  Strata zapisana w checkpoincie: {checkpoint['best_loss']}")
        else:
            # Jeśli checkpoint nie jest słownikiem lub nie ma klucza, załóż starą strukturę
            print("  Ostrzeżenie: Checkpoint nie zawiera klucza 'model_state_dict'. Zakładam, że plik zawiera tylko state_dict modelu.")
            state_dict = checkpoint
        # --- KONIEC KLUCZOWEJ ZMIANY ---


        # 4. Usuń potencjalne prefixy (działamy na wyodrębnionym `state_dict`)
        original_keys = list(state_dict.keys()) # Dla debugowania
        needs_prefix_check = True # Flaga do kontroli
        if all(key.startswith('base_encoder.') for key in state_dict.keys()):
            print("  Wykryto prefix 'base_encoder.', usuwanie...")
            state_dict = {k.replace('base_encoder.', '', 1): v for k, v in state_dict.items()}
        # Teraz klucze powinny zaczynać się od 'encoder.' lub bezpośrednio od cyfr
        # Sprawdź, czy celem jest wewnętrzny 'encoder' (jak w ResNetWrapper)
        target_model_for_load = None
        target_model_name = "enkodera"
        if hasattr(encoder, 'encoder') and isinstance(encoder.encoder, nn.Module):
            print("  Cel ładowania: 'encoder.encoder' (struktura z wrapperem).")
            target_model_for_load = encoder.encoder
            target_model_name = "'encoder.encoder'"
            # Jeśli state_dict ma prefix 'encoder.', usuńmy go
            if all(key.startswith('encoder.') for key in state_dict.keys()):
                print("    Wykryto prefix 'encoder.' w state_dict, usuwanie...")
                state_dict = {k.replace('encoder.', '', 1): v for k, v in state_dict.items()}
                needs_prefix_check = False # Już przetworzyliśmy prefix 'encoder.'

        else:
            print("  Cel ładowania: główny obiekt enkodera.")
            target_model_for_load = encoder
            # Sprawdź, czy state_dict nie ma 'encoder.', gdy nie powinien
            if any(key.startswith('encoder.') for key in state_dict.keys()):
                 print("  Ostrzeżenie: State_dict zawiera prefix 'encoder.', ale model docelowy go nie oczekuje.")

        # Usuń prefix 'module.', jeśli istnieje i nie sprawdzaliśmy 'encoder.'
        if needs_prefix_check and all(key.startswith('module.') for key in state_dict.keys()):
            print("  Wykryto prefix 'module.', usuwanie...")
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            # Sprawdź ponownie prefix 'encoder.', jeśli 'module.' go zawierał
            if hasattr(encoder, 'encoder') and isinstance(encoder.encoder, nn.Module) and all(key.startswith('encoder.') for key in state_dict.keys()):
                 print("    Wykryto prefix 'encoder.' po usunięciu 'module.', usuwanie...")
                 state_dict = {k.replace('encoder.', '', 1): v for k, v in state_dict.items()}


        # 5. Załaduj state_dict do odpowiedniego modułu
        if target_model_for_load is not None:
            print(f"  Ładowanie wag do {target_model_name}...")
            missing_keys, unexpected_keys = target_model_for_load.load_state_dict(state_dict, strict=False)
            print(f"  Wynik load_state_dict: Brakujące klucze: {len(missing_keys)}, Niespodziewane klucze: {len(unexpected_keys)}")
            if unexpected_keys:
                 print(f"  Niespodziewane klucze (pierwsze 5): {unexpected_keys[:5]}")
                 # Można dodać więcej debugowania, np. porównanie kluczy
                 # print("  Klucze w modelu docelowym:", list(target_model_for_load.state_dict().keys()))
                 # print("  Klucze w state_dict (po przetworzeniu):", list(state_dict.keys()))
            if missing_keys:
                 print(f"  Brakujące klucze (pierwsze 5): {missing_keys[:5]}")

            if not missing_keys and not unexpected_keys:
                 print("  Wagi załadowane pomyślnie.")
            else:
                 print("  Wagi załadowane (strict=False), wystąpiły pewne różnice w kluczach.")
        else:
             raise RuntimeError("Nie udało się zidentyfikować modelu docelowego do załadowania wag.")

    except FileNotFoundError: # Obsłuż konkretny błąd braku pliku
        print(f"BŁĄD KRYTYCZNY: Plik checkpointu nie istnieje: {args.checkpoint_path}")
        raise
    except Exception as e: # Obsłuż inne błędy ładowania
        print(f"BŁĄD KRYTYCZNY podczas ładowania wag z {args.checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 6. Zamroź parametry enkodera
    print("  Zamrażanie parametrów enkodera...")
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval() # Ustaw enkoder w tryb ewaluacji

    # 7. Przenieś na urządzenie
    encoder = encoder.to(device)
    print(f"Enkoder {args.arch} załadowany, zamrożony i przeniesiony na {device}.")
    print(f"--- Ładowanie Enkodera Zakończone ---")
    return encoder

def get_eval_dataloader(args):
    """Pobiera DataLoadery dla treningu i testu klasyfikatora liniowego."""
    # Używamy standardowych transformacji ('eval')
    transform_mode = 'eval'

    # Wspólne argumenty dla DataLoaderów ewaluacyjnych
    common_loader_args = {
        'batch_size': args.eval_batch_size,
        'num_workers': args.num_workers,
        'image_size': args.image_size,
        'download': True # Pozwól na pobieranie, jeśli dane nie istnieją
    }

    # Argumenty specyficzne dla transformacji 'eval'
    eval_transform_args_train = {'transform_mode': transform_mode}
    eval_transform_args_test = {'transform_mode': transform_mode}


    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        normalize = transforms.Normalize(mean=mean, std=std)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=args.data_dir,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        # train_loader = get_cifar10_dataloader(root=args.data_dir, train=True, **common_loader_args)
        test_loader = get_cifar10_dataloader(root=args.data_dir, train=False, transform_mode='eval', # Użyj eval transform dla test
                                             batch_size=256, num_workers=8, image_size=args.image_size)
        # train_loader = get_cifar10_dataloader(root=args.data_dir, train=True, **eval_transform_args_train, **common_loader_args)
        # test_loader = get_cifar10_dataloader(root=args.data_dir, train=False, **eval_transform_args_test, **common_loader_args)
    elif args.dataset == 'svhn':
        train_loader = get_svhn_dataloader(root=args.data_dir, split='train', **eval_transform_args_train, **common_loader_args)
        test_loader = get_svhn_dataloader(root=args.data_dir, split='test', **eval_transform_args_test, **common_loader_args)
    elif args.dataset == 'celeba':
        # Dla CelebA upewnij się, że masz odpowiednie etykiety (np. 'identity') lub atrybuty
        # Zmieniamy target_type na 'attr' jeśli klasyfikujemy atrybuty, lub 'identity' jeśli osoby
        target_type_eval = 'identity' # Lub 'attr', w zależności od celu ewaluacji
        train_loader = get_celeba_dataloader(root=args.data_dir, split='train', target_type=target_type_eval, download=False, **eval_transform_args_train, **common_loader_args)
        test_loader = get_celeba_dataloader(root=args.data_dir, split='test', target_type=target_type_eval, download=False, **eval_transform_args_test, **common_loader_args)
    elif args.dataset == 'imagenet_subset':
        train_loader = get_imagenet_subset_dataloader(root=args.imagenet_subset_path, split='train', **eval_transform_args_train, **common_loader_args)
        test_loader = get_imagenet_subset_dataloader(root=args.imagenet_subset_path, split='val', **eval_transform_args_test, **common_loader_args)
    else:
        raise ValueError(f"Nieznany dataset: {args.dataset}")

    print(f"Załadowano dane treningowe i testowe dla ewaluacji: {args.dataset} (transformacja: {transform_mode})")
    return train_loader, test_loader

def get_eval_optimizer(linear_classifier, args):
    """Tworzy optymalizator dla klasyfikatora liniowego."""
    if args.eval_optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            linear_classifier.parameters(), # TYLKO parametry klasyfikatora!
            lr=args.eval_lr,
            weight_decay=args.eval_weight_decay
        )
    elif args.eval_optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            linear_classifier.parameters(), # TYLKO parametry klasyfikatora!
            lr=args.eval_lr,
            momentum=0.9,
            weight_decay=args.eval_weight_decay
        )
    else:
        raise ValueError(f"Nieznany optymalizator ewaluacyjny: {args.eval_optimizer}")
    print(f"Używany optymalizator dla głowy liniowej: {optimizer.__class__.__name__}")
    return optimizer

def get_eval_scheduler(optimizer, args):
    """Tworzy harmonogram uczenia dla klasyfikatora liniowego."""
    if args.eval_scheduler == 'step':
        try:
            decay_epochs = list(map(int, args.eval_lr_decay_epochs.split(',')))
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=args.eval_lr_decay_rate)
            print(f"Używany harmonogram ewaluacyjny: MultiStepLR (epoki: {decay_epochs}, gamma: {args.eval_lr_decay_rate})")
        except ValueError:
             print(f"Błąd parsowania eval_lr_decay_epochs: {args.eval_lr_decay_epochs}. Używam StepLR.")
             scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.eval_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.eval_epochs, eta_min=0)
        print("Używany harmonogram ewaluacyjny: CosineAnnealingLR")
    elif args.eval_scheduler == 'none':
        scheduler = None
        print("Nie używam harmonogramu uczenia dla ewaluacji.")
    else:
        raise ValueError(f"Nieznany harmonogram ewaluacyjny: {args.eval_scheduler}")
    return scheduler

def calculate_accuracy(outputs, targets):
    """Oblicza dokładność Top-1."""
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total

# ----------- Main Evaluation Logic -----------

def main(args):
    """Główna funkcja sterująca ewaluacją liniową."""
    # Ustawienie urządzenia i ziarna losowości
    device = torch.device(args.device)
    set_seed(args.seed)
    print(f"Używane urządzenie: {device}")
    print(f"Ziarno losowości: {args.seed}")

    # Ładowanie zamrożonego enkodera
    encoder = load_encoder(args, device)
    encoder_output_dim = encoder.output_dim # Pobierz wymiar z wrappera

    # Definicja klasyfikatora liniowego
    # linear_classifier = nn.Linear(encoder_output_dim, args.num_classes).to(device)
    # print(f"Utworzono klasyfikator liniowy: {encoder_output_dim} -> {args.num_classes}")

    num_classes = args.num_classes

    # --- Tworzenie Klasyfikatora MLP ---
    # Określ wymiary warstw ukrytych (można je dodać do argumentów argparse)
    hidden_dim1 = 1024 # Przykład
    hidden_dim2 = 512  # Przykład
    dropout_prob = 0.5 # Przykład

    # Sprawdź, czy wymiary ukryte nie są większe od wejścia/wyjścia, jeśli nie ma sensu
    # hidden_dim1 = min(hidden_dim1, encoder_output_dim)
    # hidden_dim2 = min(hidden_dim2, hidden_dim1)


    linear_classifier = MLPClassifier( # Zamiast linear_classifier
        input_dim=encoder_output_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        output_dim=num_classes,
        dropout_prob=dropout_prob
    ).to(device)

    # Ładowanie danych (treningowych i testowych z etykietami)
    train_loader, test_loader = get_eval_dataloader(args)

    # Optymalizator i scheduler dla klasyfikatora liniowego
    optimizer = get_eval_optimizer(linear_classifier, args)
    scheduler = get_eval_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss().to(device)

    # Pętla treningowa dla klasyfikatora liniowego
    print(f"Rozpoczynanie treningu klasyfikatora liniowego na {args.eval_epochs} epok...")
    start_training_time = time.time()

    for epoch in range(args.eval_epochs):
        epoch_start_time = time.time()
        linear_classifier.train() # Ustaw klasyfikator w tryb treningowy
        encoder.eval()           # Enkoder zawsze w trybie eval

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for step, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            # Ekstrakcja cech (bez obliczania gradientów dla enkodera)
            with torch.no_grad():
                features = encoder(images) # Kształt: [B, encoder_output_dim]

            # Forward pass przez klasyfikator liniowy
            outputs = linear_classifier(features) # Kształt: [B, num_classes]

            # Obliczenie straty i backward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Statystyki treningowe
            total_loss += loss.item() * images.size(0)
            total_correct += (torch.max(outputs.data, 1)[1] == targets).sum().item()
            total_samples += images.size(0)

            # Logowanie postępu treningu
            if (step + 1) % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Eval Epoch [{epoch+1}/{args.eval_epochs}] | Step [{step+1}/{len(train_loader)}] | '
                      f'Loss: {loss.item():.4f} | LR: {current_lr:.6f}')

        # Koniec epoki treningowej
        avg_epoch_loss = total_loss / total_samples
        epoch_accuracy = 100 * total_correct / total_samples
        print(f'--- Koniec Epoki Treningowej [{epoch+1}/{args.eval_epochs}] ---')
        print(f'Średnia strata treningowa głowy liniowej: {avg_epoch_loss:.4f}')
        print(f'Dokładność treningowa głowy liniowej: {epoch_accuracy:.2f}%')

        # Krok harmonogramu
        if scheduler:
            scheduler.step()

    training_time = time.time() - start_training_time
    print(f"--- Trening klasyfikatora liniowego zakończony ---")
    print(f"Czas treningu głowy liniowej: {training_time:.2f}s")

    # Ewaluacja na zbiorze testowym
    print("Rozpoczynanie ewaluacji na zbiorze testowym...")
    linear_classifier.eval() # Ustaw klasyfikator w tryb ewaluacji
    encoder.eval()           # Enkoder pozostaje w trybie ewaluacji

    total_correct_test = 0
    total_samples_test = 0

    with torch.no_grad(): # Wyłącz obliczanie gradientów dla całej ewaluacji
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)

            # Ekstrakcja cech
            features = encoder(images)

            # Predykcje klasyfikatora liniowego
            outputs = linear_classifier(features)

            # Obliczanie dokładności
            total_correct_test += (torch.max(outputs.data, 1)[1] == targets).sum().item()
            total_samples_test += images.size(0)

    final_accuracy = 100 * total_correct_test / total_samples_test
    print("-" * 50)
    print(f"Końcowa Dokładność (Linear Probing) na Zbiorze Testowym {args.dataset}:")
    print(f">> Top-1 Accuracy: {final_accuracy:.2f}% <<")
    print("-" * 50)

# ----------- Entry Point -----------

if __name__ == "__main__":
    # Pobierz aktualną datę i godzinę
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=2))) # CEST UTC+2
    print(f"Skrypt ewaluacyjny uruchomiony: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    args = parse_arguments()
    print("--- Konfiguracja Ewaluacji ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-----------------------------")
    main(args)