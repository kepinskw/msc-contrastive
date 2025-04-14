# datasets/cifar10.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from augmentation.transforms import SimCLRTransform, StandardTransform, BasicAugmentation
from PIL import Image

class CIFAR10Contrastive(CIFAR10):
    """
    Wrapper for CIFAR10 dataset to handle different transform modes needed
    for contrastive learning (e.g., SimCLR returns two views).
    """
    def __init__(self, root, train=True, transform_mode='simclr', download=True, image_size=32):
        # Initialize with a dummy transform first
        super().__init__(root, train=train, transform=None, target_transform=None, download=download)

        self.transform_mode = transform_mode
        if transform_mode == 'simclr':
            self.transform = SimCLRTransform(image_size=image_size, dataset='cifar10')
        elif transform_mode == 'eval':
            self.transform = StandardTransform(image_size=image_size, dataset='cifar10', train=train)
        elif transform_mode == 'basic_augment':
             self.transform = BasicAugmentation(image_size=image_size, dataset='cifar10')
        else: # 'none' or other cases
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor() # Minimal transform if needed raw
            ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img) # Convert numpy array to PIL Image

        if self.transform is not None:
            processed_img = self.transform(img)
        else:
            processed_img = img # Should usually have a transform

        # SimCLR transform returns two views, others return one
        if self.transform_mode == 'simclr':
            # processed_img is already a tuple (view1, view2) from SimCLRTransform
            return processed_img, target
        else:
            # Return the single transformed view
            return processed_img, target


def get_cifar10_dataloader(root='./data', batch_size=128, num_workers=4, transform_mode='simclr', train=True, download=True, image_size=32):
    """
    Helper function to get CIFAR10 DataLoader.
    Args:
        root (str): Path to dataset directory.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes for data loading.
        transform_mode (str): 'simclr' (two views), 'eval' (standard eval transforms),
                              'basic_augment' (simple train augment), 'none' (minimal).
        train (bool): Load training or test set.
        download (bool): Download dataset if not present.
        image_size (int): Target image size.
    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = CIFAR10Contrastive(
        root=root,
        train=train,
        transform_mode=transform_mode,
        download=download,
        image_size=image_size
    )

    # Disable shuffling for the test set
    shuffle = train

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # Improves data transfer speed to GPU
        drop_last=train # Drop last incomplete batch during training
    )
    return dataloader

# Example Usage (można umieścić w bloku if __name__ == "__main__":)
if __name__ == '__main__':
    # Get SimCLR training loader
    train_loader_simclr = get_cifar10_dataloader(transform_mode='simclr', train=True)
    # Get standard evaluation test loader
    test_loader_eval = get_cifar10_dataloader(transform_mode='eval', train=False)

    # Iterate over one batch for inspection
    for (views, labels) in train_loader_simclr:
        print("SimCLR mode:")
        print("Views tuple length:", len(views))
        print("View 1 shape:", views[0].shape) # Shape: [batch_size, C, H, W]
        print("View 2 shape:", views[1].shape)
        print("Labels shape:", labels.shape)
        break

    for (images, labels) in test_loader_eval:
        print("\nEval mode:")
        print("Images shape:", images.shape) # Shape: [batch_size, C, H, W]
        print("Labels shape:", labels.shape)
        break

def get_cifar10_dataset(root='./data', transform_mode='simclr', train=True, download=True, image_size=32):
    """
    Zwraca instancję CIFAR10Contrastive (dataset), bez DataLoadera.
    Przydatne np. dla TripletDataset.
    """
    return CIFAR10Contrastive(
        root=root,
        train=train,
        transform_mode=transform_mode,
        download=download,
        image_size=image_size
    )
