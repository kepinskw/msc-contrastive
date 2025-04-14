# datasets/imagenet_subset.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from augmentation.transforms import SimCLRTransform, StandardTransform, BasicAugmentation
import os
from PIL import Image

class ImageNetSubsetContrastive(ImageFolder):
    """
    Wrapper for ImageNet subset using ImageFolder structure.
    Assumes data is organized as root/split/class/image.jpg
    (e.g., data/imagenet_100/train/n01440764/...)
    """
    def __init__(self, root, split='train', transform_mode='simclr', image_size=224):
        data_path = os.path.join(root, split)
        if not os.path.isdir(data_path):
             raise ValueError(f"ImageNet subset path not found or not directory: {data_path}. Ensure it follows ImageFolder structure.")

        # Initialize with a dummy transform first
        super().__init__(data_path, transform=None)

        self.transform_mode = transform_mode
        is_train = (split == 'train')

        if transform_mode == 'simclr':
            self.transform = SimCLRTransform(image_size=image_size, dataset='imagenet')
        elif transform_mode == 'eval':
            # ImageNet eval uses specific resize/crop
            self.transform = StandardTransform(image_size=image_size, dataset='imagenet', train=is_train)
        elif transform_mode == 'basic_augment':
             self.transform = BasicAugmentation(image_size=image_size, dataset='imagenet')
        else:
            self.transform = transforms.Compose([ # Minimal transform for raw loading
                 transforms.Resize(256),
                 transforms.CenterCrop(image_size),
                 transforms.ToTensor()
            ])

    def __getitem__(self, index):
        # ImageFolder __getitem__ returns (sample, target) where sample is PIL Image
        img, target = super().__getitem__(index)

        if self.transform is not None:
            processed_img = self.transform(img)
        else:
            processed_img = img

        if self.transform_mode == 'simclr':
            return processed_img, target
        else:
            return processed_img, target

def get_imagenet_subset_dataloader(root, batch_size=128, num_workers=4, transform_mode='simclr', split='train', image_size=224):
    """
    Helper function to get ImageNet subset DataLoader.
    Args:
        root (str): Path to the ROOT directory containing 'train' and 'val' (or 'test') subfolders
                     structured for ImageFolder (e.g., root='path/to/imagenet_100').
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        transform_mode (str): 'simclr', 'eval', 'basic_augment', 'none'.
        split (str): 'train' or 'val' (or 'test' depending on your folder name).
        image_size (int): Target image size (usually 224 for ImageNet).
    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = ImageNetSubsetContrastive(
        root=root,
        split=split,
        transform_mode=transform_mode,
        image_size=image_size
    )
    shuffle = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle
    )
    return dataloader


def get_imagenet_subset_dataset(root, transform_mode='simclr', split='train', image_size=224):
    """
    Helper function to get ImageNet subset DataLoader.
    Args:
        root (str): Path to the ROOT directory containing 'train' and 'val' (or 'test') subfolders
                     structured for ImageFolder (e.g., root='path/to/imagenet_100').
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        transform_mode (str): 'simclr', 'eval', 'basic_augment', 'none'.
        split (str): 'train' or 'val' (or 'test' depending on your folder name).
        image_size (int): Target image size (usually 224 for ImageNet).
    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = ImageNetSubsetContrastive(
        root=root,
        split=split,
        transform_mode=transform_mode,
        image_size=image_size
    )
    return dataset
# Example Usage
# if __name__ == '__main__':
#     # IMPORTANT: Replace with the ACTUAL path to your ImageNet subset directory
#     imagenet_root = '/path/to/your/imagenet_subset' # e.g., /data/imagenet_100
#     if os.path.isdir(imagenet_root):
#         try:
#             train_loader_simclr = get_imagenet_subset_dataloader(imagenet_root, transform_mode='simclr', split='train')
#             print(f"Loaded ImageNet subset from {imagenet_root}")
#             # Iterate...
#             for (views, labels) in train_loader_simclr:
#                 print("SimCLR mode (ImageNet):")
#                 print("View 1 shape:", views[0].shape)
#                 print("Labels shape:", labels.shape)
#                 break
#         except Exception as e:
#              print(f"Error loading ImageNet subset: {e}")
#     else:
#         print(f"ImageNet subset directory not found: {imagenet_root}. Skipping example.")