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

# Importuj komponenty z projektu
try:
    from datasets.cifar10 import get_cifar10_dataloader
    from datasets.svhn import get_svhn_dataloader
    from datasets.celeba import get_celeba_dataloader
    from datasets.imagenet_subset import get_imagenet_subset_dataloader
    from models.resnet_base import get_resnet_encoder
    from models.projection_head import ProjectionHead
    from methods.simclr import SimCLRNet
    from methods.siamese import SiameseNet
    from methods.triplet_net import TripletNet
    from losses.nt_xent import NTXentLoss
    from losses.contrastive import ContrastiveLoss
    from losses.triplet import TripletLoss
    from models import get_resnet_encoder
except ImportError as e:
    print(f"Błąd importu w main_eval.py: {e}")
    print("Upewnij się, że uruchamiasz skrypt z głównego katalogu projektu lub ścieżka Pythona jest poprawnie ustawiona.")
    sys.exit(1)

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
    """Ładuje pre-trenowany enkoder bazowy, zamraża go i przenosi na urządzenie."""
    print(f"Ładowanie enkodera bazowego: {args.arch}")
    encoder = get_resnet_encoder(name=args.arch, pretrained=False) # Zawsze ładujemy bez wag pre-trenowanych z torchvision

    # Załaduj zapisane wagi z checkpointu SSL
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Plik checkpointu nie znaleziony: {args.checkpoint_path}")

    try:
        state_dict = torch.load(args.checkpoint_path, map_location='cpu') # Załaduj najpierw na CPU
        # Jeśli state_dict zawiera prefix 'module.' (z DataParallel), usuń go
        if all(key.startswith('module.') for key in state_dict.keys()):
            print("Wykryto prefix 'module.' w checkpoincie, usuwanie...")
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        # Jeśli state_dict zawiera prefix 'encoder.' (z ResNetWrapper), usuń go
        if all(key.startswith('encoder.') for key in state_dict.keys()):
             print("Wykryto prefix 'encoder.' w checkpoincie, usuwanie...")
             state_dict = {k.replace('encoder.', '', 1): v for k, v in state_dict.items()}

        # Dopasuj klucze, jeśli model był opakowany inaczej
        # Czasem trzeba załadować do `encoder.encoder` jeśli używamy ResNetWrapper
        if hasattr(encoder, 'encoder') and isinstance(encoder.encoder, nn.Sequential):
             missing_keys, unexpected_keys = encoder.encoder.load_state_dict(state_dict, strict=False) # Użyj strict=False dla elastyczności
             if unexpected_keys: print(f"Ostrzeżenie: Niespodziewane klucze w checkpoincie: {unexpected_keys}")
             if missing_keys: print(f"Ostrzeżenie: Brakujące klucze w modelu: {missing_keys}")
             print(f"Załadowano wagi do 'encoder.encoder' dla {args.arch}.")
        else:
             encoder.load_state_dict(state_dict)
             print(f"Załadowano wagi bezpośrednio do {args.arch}.")

    except Exception as e:
        print(f"Błąd podczas ładowania wag z {args.checkpoint_path}: {e}")
        raise

    # Zamroź parametry enkodera
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval() # Ustaw enkoder w tryb ewaluacji
    encoder = encoder.to(device)
    print(f"Enkoder {args.arch} załadowany, zamrożony i przeniesiony na {device}.")
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
    eval_transform_args_train = {'transform_mode': transform_mode, 'train': True}
    eval_transform_args_test = {'transform_mode': transform_mode, 'train': False}


    if args.dataset == 'cifar10':
        train_loader = get_cifar10_dataloader(root=args.data_dir, train=True, **eval_transform_args_train, **common_loader_args)
        test_loader = get_cifar10_dataloader(root=args.data_dir, train=False, **eval_transform_args_test, **common_loader_args)
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
    linear_classifier = nn.Linear(encoder_output_dim, args.num_classes).to(device)
    print(f"Utworzono klasyfikator liniowy: {encoder_output_dim} -> {args.num_classes}")

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