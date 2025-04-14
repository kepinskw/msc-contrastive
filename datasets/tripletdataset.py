import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict

class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        """
        base_dataset: instancja datasetu zwracającego (image, label)
        """
        self.base_dataset = base_dataset
        self.data_by_class = defaultdict(list)
        
        # Organizujemy indeksy danych według klas
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            self.data_by_class[label].append(idx)
        
        self.labels = list(self.data_by_class.keys())
        
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        anchor_img, anchor_label = self.base_dataset[idx]
        
        # Dobieramy positive (inna próbka z tej samej klasy)
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.data_by_class[anchor_label])
        positive_img, _ = self.base_dataset[positive_idx]

        # Dobieramy negative (próbka z innej klasy)
        negative_label = random.choice([label for label in self.labels if label != anchor_label])
        negative_idx = random.choice(self.data_by_class[negative_label])
        negative_img, _ = self.base_dataset[negative_idx]

        return (anchor_img, positive_img, negative_img)
