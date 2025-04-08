# datasets/celeba.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from augmentation.transforms import SimCLRTransform, StandardTransform, BasicAugmentation
from PIL import Image

class CelebAContrastive(CelebA):
    """Wrapper for CelebA dataset."""
    def __init__(self, root, split='train', target_type='attr', transform_mode='simclr', download=False, image_size=128): # CelebA images are larger
        # Note: CelebA download via torchvision might be unreliable. Often requires manual download.
        # Set download=False and ensure data is in 'root/celeba'
        super().__init__(root, split=split, target_type=target_type, transform=None, target_transform=None, download=download)

        self.transform_mode = transform_mode
        is_train = (split == 'train')

        if transform_mode == 'simclr':
            self.transform = SimCLRTransform(image_size=image_size, dataset='celeba') # Use default norm
        elif transform_mode == 'eval':
             # Adjust eval size if needed, e.g., 128 or 224
            self.transform = StandardTransform(image_size=image_size, dataset='celeba', train=is_train)
        elif transform_mode == 'basic_augment':
             self.transform = BasicAugmentation(image_size=image_size, dataset='celeba')
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(178), # CelebA specific cropping often used
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        # Target transform can be added if needed for attributes/identity processing

    def __getitem__(self, index):
        # CelebA returns PIL Image directly
        img, target = super().__getitem__(index)

        if self.transform is not None:
            processed_img = self.transform(img)
        else:
            processed_img = img

        if self.transform_mode == 'simclr':
            return processed_img, target # Target might be attributes or identity
        else:
            return processed_img, target


def get_celeba_dataloader(root='./data', batch_size=128, num_workers=4, transform_mode='simclr', split='train', target_type='attr', download=False, image_size=128):
    """Helper function to get CelebA DataLoader."""
    dataset = CelebAContrastive(
        root=root,
        split=split,
        target_type=target_type,
        transform_mode=transform_mode,
        download=download,
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

# Example Usage
# if __name__ == '__main__':
#     # Ensure you have the CelebA dataset downloaded and extracted in ./data/celeba
#     # It contains Align&Cropped Images, attribute lists, identity lists etc.
#     try:
#         train_loader_simclr = get_celeba_dataloader(transform_mode='simclr', split='train', image_size=128)
#         print("Loaded CelebA")
#         # Iterate...
#     except RuntimeError as e:
#         print(f"Could not load CelebA. Ensure data is in ./data/celeba: {e}")