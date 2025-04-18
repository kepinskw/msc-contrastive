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

# (Opcjonalnie) Import bardziej zaawansowanych optymalizatorów, np. LARS
# from torch.optim import LARS # uzywany w pracy

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

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Ścieżka do checkpointu, od którego wznowić trening (domyślnie: nie wznawiaj)')
    
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
        transform_mode = 'simclr'
    elif args.method == 'siamese':
        transform_mode = 'siamese'
    elif args.method == 'triplet':
        transform_mode = 'triplet'
    else:
        print(f"Ostrzeżenie: Nieznana metoda '{args.method}' w get_dataloader. Używam transform_mode='eval'.")
        transform_mode = 'eval' # LUB 'basic_augment', zależy co tam było

    common_loader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'transform_mode': transform_mode,
        'image_size': args.image_size
    }

    common_dataset_args = {
    'transform_mode': transform_mode,
    'image_size': args.image_size,
    'train': True,
    'download': True
    }

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

        test_loader = None
    elif args.dataset == 'svhn':

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

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
        train_dataset = datasets.SVHN(root=args.data_dir,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        test_loader = None

        # train_loader = get_svhn_dataloader(root=args.data_dir, split='train', **common_loader_args)
        # test_loader = get_svhn_dataloader(root=args.data_dir, split='test', transform_mode='eval',
        #                                     batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    elif args.dataset == 'celeba':
        train_loader = get_celeba_dataloader(root=args.data_dir, train=True, target_type='identity', # Użyjmy identity do ew. tworzenia par
                                             download=False, **common_loader_args)
        test_loader = get_celeba_dataloader(root=args.data_dir, train=False, target_type='identity', transform_mode='eval',
                                             download=False, batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    elif args.dataset == 'imagenet_subset':
        train_loader = get_imagenet_subset_dataloader(root=args.imagenet_subset_path, split='train', **common_loader_args)
        test_loader = get_imagenet_subset_dataloader(root=args.imagenet_subset_path, split='val', transform_mode='eval', # Zazwyczaj 'val' dla ImageNet
                                                    batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    else:
        raise ValueError(f"Nieznany dataset: {args.dataset}")



    print(f"Załadowano dane treningowe dla: {args.dataset} z transformacją: {transform_mode}")
    return train_loader, test_loader

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
    # elif args.optimizer.lower() == 'lars':
    #     optimizer = LARS(
    #         model.parameters(),
    #         lr=args.lr,
    #     )


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

    for step, (images,labels) in enumerate(dataloader):#for step, batch_data in enumerate(dataloader):
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
            
            try:
                img1, img2, pair_label = batch_data
                img1, img2, pair_label = img1.to(device), img2.to(device), pair_label.to(device)
            except ValueError:
                 print(f"BŁĄD: DataLoader dla metody '{args.method}' nie zwraca oczekiwanych par (img1, img2, label). Zwrócono: {type(batch_data)}")
                 
            # Forward pass dla Siamese
            emb1, emb2 = model(img1, img2)

            # Oblicz stratę ContrastiveLoss
            loss = criterion(emb1, emb2, pair_label)

        elif args.method == 'triplet':
            #unsuprevised
            images = torch.cat([images[0],images[1]],dim=0).to(device)
            bsz = labels.shape[0]
            
            features = model(images) # Forward pass
            f1,f2 = torch.split(features, [bsz,bsz], dim=0)#.to(device)
            features = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)], dim=1).to(device)
            
            anchor_batch = f1.to(device)
            positive_batch = f2.to(device)
            indices = torch.arange(bsz).to(device)
            # shifted_indices = torch.roll(indices, shifts=1, dims=0)
            # negative_batch = f2[shifted_indices]
            random_shifts = torch.randint(1, bsz, (bsz,)).to(device)
            negative_indices = (indices + random_shifts) % bsz 
            negative_batch = f2[negative_indices].to(device)
            loss= criterion(anchor_batch, positive_batch, negative_batch)
            loss= criterion(anchor_batch, positive_batch, negative_batch)

            # anchor, positive, negative = batch_data
            # anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            # # Forward pass dla Triplet (wywołany 3 razy)
            # emb_a = model(anchor)
            # emb_p = model(positive)
            # emb_n = model(negative)

            # # Oblicz stratę TripletLoss
            # loss = criterion(emb_a, emb_p, emb_n)

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

    writer = SummaryWriter(log_dir=os.path.join('logs', args.run_name))

    # Ładowanie danych
    train_loader, _ = get_dataloader(args) # Ignorujemy test_loader w treningu SSL

    # Inicjalizacja modelu, funkcji straty, optymalizatora i harmonogramu
    model = get_model(args).to(device)
    criterion = get_loss(args, device)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    start_epoch = 0 
    best_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Ładowanie checkpointu '{args.resume}'")
            try:
                # Załaduj checkpoint na CPU najpierw, aby uniknąć problemów z GPU
                checkpoint = torch.load(args.resume, map_location='cpu')

                # Załaduj stan modelu
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                    # Obsługa prefixu 'module.' jeśli model był trenowany z nn.DataParallel
                    if all(key.startswith('module.') for key in model_state_dict.keys()):
                         print("  Wykryto prefix 'module.' w checkpoincie modelu, usuwanie...")
                         model_state_dict = {k.replace('module.', '', 1): v for k, v in model_state_dict.items()}

                    # Załaduj stan do bieżącego modelu
                    try:
                         model.load_state_dict(model_state_dict, strict=True)
                         print("  Pomyślnie załadowano stan modelu.")
                    except RuntimeError as e:
                         print(f"  Ostrzeżenie: Nie udało się załadować stanu modelu (strict=True): {e}. Sprawdź architekturę.")
                         # Możesz spróbować strict=False, ale bądź ostrożny:
                         # model.load_state_dict(model_state_dict, strict=False)
                         # print("  Załadowano stan modelu z strict=False (sprawdź ostrzeżenia).")
                else:
                    print("  Ostrzeżenie: Brak 'model_state_dict' w checkpoincie.")

                # Załaduj stan optymalizatora
                if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        optimizer.param_groups[0]['lr'] = args.lr#optimizer.param_groups[0]['initial_lr']
                        print("  Pomyślnie załadowano stan optymalizatora.")
                        # Ważne: Przenieś stan optymalizatora na właściwe urządzenie, jeśli to konieczne
                        for state in optimizer.state.values():
                             for k, v in state.items():
                                 if isinstance(v, torch.Tensor):
                                     state[k] = v.to(device)
                    except Exception as e:
                        print(f"  Ostrzeżenie: Nie udało się załadować stanu optymalizatora: {e}")
                else:
                    print("  Ostrzeżenie: Brak 'optimizer_state_dict' w checkpoincie lub optimizer=None.")

                # Załaduj stan schedulera (jeśli istnieje)
                if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                     try:
                         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                         print("  Pomyślnie załadowano stan schedulera.")
                     except Exception as e:
                          print(f"  Ostrzeżenie: Nie udało się załadować stanu schedulera: {e}")
                # else: print("  Info: Brak 'scheduler_state_dict' w checkpoincie lub scheduler=None.") # Mniej istotne ostrzeżenie

                # Załaduj numer epoki (zapisujemy numer *zakończonej* epoki)
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                    print(f"  Wznawianie treningu od epoki: {start_epoch + 1}")
                else:
                    print("  Ostrzeżenie: Brak informacji o epoce w checkpoincie. Zaczynam od epoki 0.")

                # Załaduj inne zapisane wartości (np. najlepsza strata)
                if 'best_loss' in checkpoint:
                    best_loss = checkpoint['best_loss']
                    print(f"  Załadowano najlepszą stratę: {best_loss:.4f}")


                print(f"=> Pomyślnie załadowano informacje z checkpointu '{args.resume}'")

            except Exception as e:
                print(f"BŁĄD: Nie udało się załadować checkpointu '{args.resume}': {e}")
                print("Rozpoczynanie treningu od początku.")
                start_epoch = 0 # Resetuj na wszelki wypadek
        else:
            print(f"OSTRZEŻENIE: Podany plik checkpointu '{args.resume}' nie istnieje!")
            print("Rozpoczynanie treningu od początku.")
    else:
        print("Nie podano argumentu --resume, rozpoczynanie treningu od początku.")


    # Główna pętla treningowa
    args.epochs += start_epoch
    print(f"Rozpoczynanie treningu na {args.epochs} epok...")
    start_training_time = time.time()

    
    
    for epoch in range(start_epoch, args.epochs):
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
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Zapisywanie checkpointu
        current_epoch_display = epoch + 1

        # Zapisujemy tylko ENKODER BAZOWY, bo to on jest używany do ewaluacji downstream
        if hasattr(model, 'base_encoder'):
            encoder_state_dict = {
                        'epoch': current_epoch_display,
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_loss': best_loss if best_loss else avg_train_loss, # Jeśli śledzisz
                        # 'args': args
                    }

        elif isinstance(model, SimCLRNet): # SimCLRNet ma enkoder jako atrybut
            encoder_state_dict = {
                        'epoch': current_epoch_display,
                        'model_state_dict': model.base_encoder.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_loss': best_loss if best_loss else avg_train_loss, # Jeśli śledzisz
                        # 'args': args
                    }
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
            encoder_state_dict[best_loss] = best_loss
            best_checkpoint_file = os.path.join(checkpoint_path, 'best_encoder.pth')
            torch.save(encoder_state_dict, best_checkpoint_file)
            print(f"Zapisano nowy najlepszy enkoder (strata: {best_loss:.4f}) do {best_checkpoint_file}")

    total_training_time = time.time() - start_training_time
    print(f"--- Trening zakończony ---")
    print(f"Całkowity czas treningu: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
    print(f"Najlepsza strata treningowa: {best_loss:.4f}")
    print(f"Ostatni enkoder zapisany w: {last_checkpoint_file}")
    print(f"Najlepszy enkoder zapisany w: {best_checkpoint_file}")

    writer.close()

# ----------- Entry Point -----------

if __name__ == "__main__":
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=2))) # CEST UTC+2
    print(f"Skrypt uruchomiony: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    args = parse_arguments()
    print("--- Konfiguracja Uruchomienia ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-------------------------------")
    main(args)