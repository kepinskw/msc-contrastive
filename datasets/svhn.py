# datasets/svhn.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import SVHN
from augmentation.transforms import SimCLRTransform, StandardTransform, BasicAugmentation
from PIL import Image # SVHN needs explicit conversion sometimes
from datasets.wrappers import TripletDataset, SiameseDataset, SimCLRDataset
import numpy as np
class SVHNContrastive(SVHN):
    """Wrapper for SVHN dataset."""
    def __init__(self, root, split='train', transform_mode='simclr', download=True, image_size=32):
        super().__init__(root, split=split, transform=None, target_transform=None, download=download)

        self.transform_mode = transform_mode
        is_train = (split == 'train') # Needed for StandardTransform train/test mode

        if transform_mode == 'simclr':
            self.transform = SimCLRTransform(image_size=image_size, dataset='svhn') # Use default norm for now
        elif transform_mode == 'eval':
            self.transform = StandardTransform(image_size=image_size, dataset='svhn', train=is_train)
        elif transform_mode == 'basic_augment':
             self.transform = BasicAugmentation(image_size=image_size, dataset='svhn')
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        # SVHN data is returned as (H, W, C) numpy uint8 array
        img, target = self.data[index], int(self.labels[index])
        # Convert to (C, H, W) and then to PIL Image: transpose(1, 2, 0)
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            processed_img = self.transform(img)
        else:
            processed_img = img

        if self.transform_mode == 'simclr':
            return processed_img, target
        else:
            return processed_img, target


def get_svhn_dataloader(root='./data', batch_size=128, num_workers=4, transform_mode='simclr', split='train', download=True, image_size=32):
    """
    Helper function to get SVHN DataLoader, obsługująca różne tryby.
    (Docstring arguments jak poprzednio)
    """
    dataset = None # Zmienna na finalny obiekt Dataset
    # split = 'train' if train else 'test'
    # 1. Wybierz i przygotuj dataset na podstawie transform_mode
    if transform_mode == 'simclr':
        # Używa SVHNContrastive, która tworzy SimCLRTransform wewnątrz
        print(f"INFO: Tworzenie datasetu SVHNContrastive dla trybu: {transform_mode}")
        dataset = SimCLRDataset(root=root, split='train', download=download, transform_mode=transform_mode, image_size=image_size)

    elif transform_mode == 'eval':
        # Używa SVHNContrastive, która tworzy StandardTransform wewnątrz
        print(f"INFO: Tworzenie datasetu SVHNContrastive dla trybu: {transform_mode}")
        dataset = SVHNContrastive(root=root, split='train', download=download, transform_mode=transform_mode, image_size=image_size)

    elif transform_mode == 'basic_augment':
        # Używa SVHNContrastive, która tworzy BasicAugmentation wewnątrz
        print(f"INFO: Tworzenie datasetu SVHNContrastive dla trybu: {transform_mode}")
        dataset = SVHNContrastive(root=root, split='train', download=download, transform_mode=transform_mode, image_size=image_size)

    elif transform_mode == 'siamese':
        print(f"INFO: Tworzenie datasetu SiameseDatasetWrapper dla trybu: {transform_mode}")
        transform_siamese = BasicAugmentation(image_size=image_size, dataset='SVHN')
        base_raw_dataset = SVHN(root=root, split='train', download=download, transform=None)
        
        dataset = SiameseDataset(base_dataset=base_raw_dataset, transform=transform_siamese)
        
    elif transform_mode == 'triplet':
        print(f"INFO: Tworzenie datasetu Triplet dla trybu: {transform_mode}")
        transform_triplet = BasicAugmentation(image_size=image_size, dataset='SVHN')
        base_raw_dataset = SVHN(root=root, split='train', download=download, transform=None)
    
        dataset = TripletDataset(base_dataset=base_raw_dataset, transform=transform_triplet)


    elif transform_mode == 'none':
        # Używa SVHNContrastive, która tworzy minimalną transformację wewnątrz
        print(f"INFO: Tworzenie datasetu SVHNContrastive dla trybu: {transform_mode}")
        dataset = SVHNContrastive(root=root, split='train', download=download, transform_mode=transform_mode, image_size=image_size)

    else:
         raise ValueError(f"Nieznany transform_mode: '{transform_mode}'")

    # Sprawdzenie, czy dataset został utworzony
    if dataset is None:
         # Teoretycznie nie powinno się zdarzyć przy poprawnej obsłudze wszystkich trybów
         raise RuntimeError(f"Dataset nie został zainicjalizowany dla transform_mode='{transform_mode}'.")
    train = True if split == 'train' else False

    # 2. Utwórz DataLoader
    shuffle = train
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )
    print(f"INFO: Utworzono DataLoader dla trybu '{transform_mode}' (train={train})")
    return dataloader

def get_svhn_dataset(root='./data',transform_mode='simclr', split='train', download=True, image_size=32):
    """Helper function to get SVHN DataLoader."""
    dataset = SVHNContrastive(
        root=root,
        split=split,
        transform_mode=transform_mode,
        download=download,
        image_size=image_size
    )
    return dataset
# Example Usage
# if __name__ == '__main__':
#     import numpy as np # Needed for transpose
#     # Get SimCLR training loader
#     train_loader_simclr = get_svhn_dataloader(transform_mode='simclr', split='train')
#     # Get standard evaluation test loader
#     test_loader_eval = get_svhn_dataloader(transform_mode='eval', split='test')