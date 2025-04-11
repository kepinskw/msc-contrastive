# main_ssl.py

import argparse
import os
import time
import datetime
import sys

import torch
import torch.optim as optim
import torch.nn as nn
# Importuj komponenty z projektu
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

# (Opcjonalnie) Import logowania, np. TensorBoard
from torch.utils.tensorboard import SummaryWriter

# (Opcjonalnie) Import bardziej zaawansowanych optymalizatorów, np. LARS
# from torch.optim import LARS

# ----------- Argument Parsing -----------

def parse_arguments():
    """Paruje argumenty linii poleceń."""
    parser = argparse.ArgumentParser(description='Self-Supervised Learning Training Script')

    # Dataset Arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'svhn', 'celeba', 'imagenet_subset'],
                        help='Nazwa datasetu do użycia.')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Ścieżka do katalogu z danymi.')
    parser.add_argument('--imagenet_subset_path', type=str, default='/path/to/imagenet_subset',
                        help='Ścieżka do podzbioru ImageNet (jeśli używany).')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Rozmiar obrazu (np. 32 dla CIFAR/SVHN, 128/224 dla CelebA/ImageNet).')

    # Model & Method Arguments
    parser.add_argument('--method', type=str, default='simclr',
                        choices=['simclr', 'siamese', 'triplet'],
                        help='Metoda uczenia kontrastowego.')
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='Architektura enkodera bazowego (np. resnet18, resnet50).')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Wymiar wyjścia głowicy projekcyjnej (dla SimCLR).')

    # Loss Function Arguments
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperatura dla NTXentLoss (SimCLR).')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Margines dla ContrastiveLoss lub TripletLoss.')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Liczba epok treningowych.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Rozmiar batcha.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'lars'],
                        help='Optymalizator.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Początkowa szybkość uczenia (learning rate).')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Współczynnik L2 regularyzacji (weight decay).')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'step', 'cosine'],
                        help='Typ harmonogramu uczenia (learning rate scheduler).')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Współczynnik zmniejszenia LR dla StepLR')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,75,90', help='Epoki do zmniejszenia LR dla StepLR (oddzielone przecinkami)')


    # System Arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Urządzenie do treningu (cuda/cpu).')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Liczba wątków roboczych dla DataLoader.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Ziarno losowości dla reprodukowalności.')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Co ile kroków logować postęp.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_ssl',
                        help='Katalog do zapisywania checkpointów.')
    parser.add_argument('--run_name', type=str, default=f'ssl_run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Nazwa bieżącego uruchomienia (dla logów i checkpointów).')


    return parser.parse_args()

# ----------- Helper Functions -----------

def set_seed(seed):
    """Ustawia ziarno losowości dla reprodukowalności."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # numpy, random również można ustawić
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

def get_dataloader(args):
    """Pobiera odpowiedni DataLoader na podstawie argumentów."""
    # Transform mode zależy od metody
    if args.method == 'simclr':
        transform_mode = 'simclr' # Potrzebuje dwóch widoków
    else:
        # Dla Siamese/Triplet - na razie użyjmy basic_augment.
        # UWAGA: Generowanie par/tripletów nie jest tutaj zaimplementowane!
        # To wymagałoby dodatkowej logiki w DataLoaderze lub w pętli treningowej.
        transform_mode = 'basic_augment'

    common_loader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'transform_mode': transform_mode,
        'image_size': args.image_size
    }

    if args.dataset == 'cifar10':
        train_loader = get_cifar10_dataloader(root=args.data_dir, train=True, **common_loader_args)
        # Walidacja SSL zazwyczaj nie jest robiona, ale można dodać test loader do ew. wizualizacji
        test_loader = get_cifar10_dataloader(root=args.data_dir, train=False, transform_mode='eval', # Użyj eval transform dla test
                                             batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    elif args.dataset == 'svhn':
        train_loader = get_svhn_dataloader(root=args.data_dir, split='train', **common_loader_args)
        test_loader = get_svhn_dataloader(root=args.data_dir, split='test', transform_mode='eval',
                                            batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    elif args.dataset == 'celeba':
        # Upewnij się, że CelebA jest pobrany i ścieżka jest poprawna
        train_loader = get_celeba_dataloader(root=args.data_dir, split='train', target_type='identity', # Użyjmy identity do ew. tworzenia par
                                             download=False, **common_loader_args)
        test_loader = get_celeba_dataloader(root=args.data_dir, split='test', target_type='identity', transform_mode='eval',
                                             download=False, batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    elif args.dataset == 'imagenet_subset':
        # Upewnij się, że ścieżka jest poprawna
        train_loader = get_imagenet_subset_dataloader(root=args.imagenet_subset_path, split='train', **common_loader_args)
        test_loader = get_imagenet_subset_dataloader(root=args.imagenet_subset_path, split='val', transform_mode='eval', # Zazwyczaj 'val' dla ImageNet
                                                    batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    else:
        raise ValueError(f"Nieznany dataset: {args.dataset}")

    print(f"Załadowano dane treningowe dla: {args.dataset} z transformacją: {transform_mode}")
    return train_loader, test_loader # Zwracamy też test_loader, chociaż nie jest używany w pętli SSL

def get_model(args):
    """Tworzy i zwraca model na podstawie argumentów."""
    # Pobierz klasę enkodera bazowego
    base_encoder_func = lambda pretrained=False: get_resnet_encoder(name=args.arch, pretrained=pretrained)

    if args.method == 'simclr':
        model = SimCLRNet(
            base_encoder_class=base_encoder_func,
            projection_dim=args.projection_dim
        )
    elif args.method == 'siamese':
        model = SiameseNet(base_encoder_class=base_encoder_func)
    elif args.method == 'triplet':
        model = TripletNet(base_encoder_class=base_encoder_func)
    else:
        raise ValueError(f"Nieznana metoda: {args.method}")

    print(f"Utworzono model dla metody: {args.method} z architekturą bazową: {args.arch}")
    return model

def get_loss(args, device):
    """Tworzy i zwraca funkcję straty na podstawie argumentów."""
    if args.method == 'simclr':
        loss_fn = NTXentLoss(
            temperature=args.temperature,
            batch_size=args.batch_size, # Przekazujemy batch_size
            device=device
        ).to(device)
    elif args.method == 'siamese':
        loss_fn = ContrastiveLoss(margin=args.margin).to(device)
    elif args.method == 'triplet':
        loss_fn = TripletLoss(margin=args.margin).to(device)
    else:
        raise ValueError(f"Nieznana metoda: {args.method}")

    print(f"Używana funkcja straty: {loss_fn.__class__.__name__}")
    return loss_fn

def get_optimizer(model, args):
    """Tworzy i zwraca optymalizator na podstawie argumentów."""
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9, # Typowa wartość dla SGD
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'lars':
        # LARS jest często zalecany dla SimCLR z dużymi batchami
        # Wymaga zewnętrznej implementacji lub nowszych wersji PyTorch
        try:
            # from torch.optim import LARS # Sprawdź, czy jest dostępny
            # optimizer = LARS(...)
            print("Ostrzeżenie: LARS nie jest standardowo dostępny, używam Adam.")
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # Fallback
        except ImportError:
            print("Ostrzeżenie: LARS nie jest dostępny, używam Adam.")
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # Fallback
    else:
        raise ValueError(f"Nieznany optymalizator: {args.optimizer}")

    print(f"Używany optymalizator: {optimizer.__class__.__name__}")
    return optimizer

def get_scheduler(optimizer, args):
    """Tworzy i zwraca harmonogram uczenia."""
    if args.scheduler == 'step':
        try:
            decay_epochs = list(map(int, args.lr_decay_epochs.split(',')))
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=decay_epochs,
                gamma=args.lr_decay_rate
            )
            print(f"Używany harmonogram: MultiStepLR (epoki: {decay_epochs}, gamma: {args.lr_decay_rate})")
        except ValueError:
            print(f"Błąd parsowania lr_decay_epochs: {args.lr_decay_epochs}. Używam StepLR z domyślnymi.")
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # Domyślny StepLR

    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs, # Liczba epok
            eta_min=0 # Minimalny LR
        )
        print("Używany harmonogram: CosineAnnealingLR")
    elif args.scheduler == 'none':
        scheduler = None
        print("Nie używam harmonogramu uczenia.")
    else:
        raise ValueError(f"Nieznany harmonogram: {args.scheduler}")

    return scheduler

# ----------- Main Training Logic -----------

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """Wykonuje jedną epokę treningową."""
    model.train() # Ustaw model w tryb treningowy
    total_loss = 0.0
    start_time = time.time()

    for step, batch_data in enumerate(dataloader):
        optimizer.zero_grad() # Wyzeruj gradienty

        # Przenieś dane na odpowiednie urządzenie
        # Rozpakowanie danych zależy od transform_mode i metody
        if args.method == 'simclr':
            # Oczekujemy (view1, view2), labels (labels są ignorowane w SSL)
            (views, _) = batch_data # Rozpakowujemy tuple (views, labels)
            view1, view2 = views[0].to(device), views[1].to(device)

            # Forward pass dla SimCLR
            h1, z1 = model(view1)
            h2, z2 = model(view2)

            # Oblicz stratę NT-Xent
            loss = criterion(z1, z2)

        elif args.method == 'siamese':
            # UWAGA: DataLoader musi dostarczać PARY (img1, img2) oraz label (0/1)
            # Obecny DataLoader zwraca (image, label) - wymaga modyfikacji!
            # Placeholder - zakładamy, że batch_data to (img1, img2, pair_label)
            try:
                img1, img2, pair_label = batch_data
                img1, img2, pair_label = img1.to(device), img2.to(device), pair_label.to(device)
            except ValueError:
                 print(f"BŁĄD: DataLoader dla metody '{args.method}' nie zwraca oczekiwanych par (img1, img2, label). Zwrócono: {type(batch_data)}")
                 # Dla demonstracji, użyjemy losowych danych - NIE UŻYWAĆ W PRAWDZIWYM TRENINGU
                 img, _ = batch_data
                 img1 = img.to(device)
                 img2 = torch.roll(img, shifts=1, dims=0).to(device) # Sztuczna druga część pary
                 pair_label = torch.randint(0, 2, (img.size(0),), device=device) # Losowe etykiety par
                 if step == 0: print("OSTRZEŻENIE: Generuję sztuczne pary dla Siamese - popraw DataLoader!")


            # Forward pass dla Siamese
            emb1, emb2 = model(img1, img2)

            # Oblicz stratę ContrastiveLoss
            loss = criterion(emb1, emb2, pair_label)

        elif args.method == 'triplet':
            # UWAGA: DataLoader musi dostarczać TRIPLETY (anchor, positive, negative)
            # Obecny DataLoader zwraca (image, label) - wymaga modyfikacji!
            # Placeholder - zakładamy, że batch_data to (anchor, positive, negative)
            try:
                anchor, positive, negative = batch_data
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            except ValueError:
                 print(f"BŁĄD: DataLoader dla metody '{args.method}' nie zwraca oczekiwanych tripletów. Zwrócono: {type(batch_data)}")
                 # Dla demonstracji - NIE UŻYWAĆ W PRAWDZIWYM TRENINGU
                 img, _ = batch_data
                 anchor = img.to(device)
                 positive = torch.roll(img, shifts=1, dims=0).to(device) # Sztuczny positive
                 negative = torch.roll(img, shifts=2, dims=0).to(device) # Sztuczny negative
                 if step == 0: print("OSTRZEŻENIE: Generuję sztuczne triplety - popraw DataLoader/Sampler!")


            # Forward pass dla Triplet (wywołany 3 razy)
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            # Oblicz stratę TripletLoss
            loss = criterion(emb_a, emb_p, emb_n)

        else:
            raise ValueError(f"Nieobsługiwana metoda w pętli treningowej: {args.method}")

        # Backward pass i krok optymalizatora
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Logowanie
        if (step + 1) % args.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            elapsed_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{args.epochs}] | Step [{step+1}/{len(dataloader)}] | '
                  f'Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed_time:.2f}s')

    avg_epoch_loss = total_loss / len(dataloader)
    return avg_epoch_loss


def main(args):
    """Główna funkcja sterująca treningiem."""
    # Ustawienie urządzenia i ziarna losowości
    device = torch.device(args.device)
    set_seed(args.seed)
    print(f"Używane urządzenie: {device}")
    print(f"Ziarno losowości: {args.seed}")

    # Przygotowanie katalogu na checkpointy
    checkpoint_path = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Checkpointy będą zapisywane w: {checkpoint_path}")

    # TODO: Inicjalizacja logowania (np. TensorBoard)
    writer = SummaryWriter(log_dir=os.path.join('logs', args.run_name))

    # Ładowanie danych
    train_loader, _ = get_dataloader(args) # Ignorujemy test_loader w treningu SSL

    # Inicjalizacja modelu, funkcji straty, optymalizatora i harmonogramu
    model = get_model(args).to(device)
    criterion = get_loss(args, device)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # Główna pętla treningowa
    print(f"Rozpoczynanie treningu na {args.epochs} epok...")
    start_training_time = time.time()

    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # Trening
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args)

        # Krok harmonogramu uczenia (jeśli używany)
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        print(f'--- Koniec Epoki [{epoch+1}/{args.epochs}] ---')
        print(f'Średnia strata treningowa: {avg_train_loss:.4f}')
        print(f'Czas epoki: {epoch_time:.2f}s')
        # TODO: Logowanie do TensorBoard/WandB
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Zapisywanie checkpointu
        # Zapisujemy tylko ENKODER BAZOWY, bo to on jest używany do ewaluacji downstream
        if hasattr(model, 'base_encoder'):
            encoder_state_dict = model.base_encoder.state_dict()
        elif isinstance(model, SimCLRNet): # SimCLRNet ma enkoder jako atrybut
             encoder_state_dict = model.base_encoder.state_dict()
        else:
             print("Ostrzeżenie: Nie można automatycznie zidentyfikować enkodera bazowego do zapisu.")
             # Spróbuj zapisać cały model, ale to mniej użyteczne dla ewaluacji SSL
             encoder_state_dict = model.state_dict()


        # Zapisz ostatni model
        last_checkpoint_file = os.path.join(checkpoint_path, 'last_encoder.pth')
        torch.save(encoder_state_dict, last_checkpoint_file)

        # Zapisz najlepszy model (na podstawie straty treningowej - uproszczenie)
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_checkpoint_file = os.path.join(checkpoint_path, 'best_encoder.pth')
            torch.save(encoder_state_dict, best_checkpoint_file)
            print(f"Zapisano nowy najlepszy enkoder (strata: {best_loss:.4f}) do {best_checkpoint_file}")

    total_training_time = time.time() - start_training_time
    print(f"--- Trening zakończony ---")
    print(f"Całkowity czas treningu: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
    print(f"Najlepsza strata treningowa: {best_loss:.4f}")
    print(f"Ostatni enkoder zapisany w: {last_checkpoint_file}")
    print(f"Najlepszy enkoder zapisany w: {best_checkpoint_file}")

    # TODO: Zamknięcie loggera
    writer.close()

# ----------- Entry Point -----------

if __name__ == "__main__":
    # Pobierz aktualną datę i godzinę
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=2))) # CEST UTC+2
    print(f"Skrypt uruchomiony: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    args = parse_arguments()
    print("--- Konfiguracja Uruchomienia ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-------------------------------")
    main(args)