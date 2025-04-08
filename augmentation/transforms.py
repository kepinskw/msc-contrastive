# augmentation/transforms.py
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps
import random

# Normalization constants (adjust if needed, especially for custom ImageNet subset)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default normalization for SVHN, CelebA might need adjustment or calculation
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarize(object):
    """Solarization augmentation"""
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, x):
        return ImageOps.solarize(x, self.threshold)

class SimCLRTransform:
    """
    Transformations for SimCLR: return two correlated views of the same image.
    Based on the original SimCLR paper and typical implementations.
    """
    def __init__(self, image_size=32, dataset='cifar10'):
        if dataset == 'cifar10':
            normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        elif dataset == 'imagenet':
             # ImageNet images are typically larger, adjust size
             image_size = 224
             normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        else: # Default for SVHN, CelebA - adjust as needed
             normalize = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) # Adjust strength as needed
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5), # SimCLR used p=0.5 for CIFAR, p=1.0 maybe for ImageNet?
            # transforms.RandomApply([Solarize()], p=0.1), # Optional: used in BYOL/SwAV
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        # Apply the transform twice to get two views
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

class StandardTransform:
    """
    Standard transformations for evaluation (e.g., linear probing).
    No data augmentation beyond basic normalization and resizing.
    """
    def __init__(self, image_size=32, dataset='cifar10', train=True):
        if dataset == 'cifar10':
            normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
            if train: # Slight augmentation for linear probing train set? Optional.
                 self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)), # Less aggressive crop
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else: # Test set: Center crop
                 self.transform = transforms.Compose([
                    transforms.Resize(int(image_size * 1.1)), # Resize slightly larger
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ])

        elif dataset == 'imagenet':
            image_size = 224
            normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            if train:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                 self.transform = transforms.Compose([
                    transforms.Resize(256), # Standard ImageNet eval resize
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
        else: # Default for SVHN, CelebA - adjust as needed
            normalize = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)), # Simple resize
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    normalize,
                ])


    def __call__(self, x):
        return self.transform(x)

# Simple Augmentation for Triplet/Siamese (can be adjusted)
class BasicAugmentation:
    def __init__(self, image_size=32, dataset='cifar10'):
        if dataset == 'cifar10':
            normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        elif dataset == 'imagenet':
             image_size = 224
             normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        else:
             normalize = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        return self.transform(x)