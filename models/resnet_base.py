# models/resnet_base.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


SUPPORTED_RESNETS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

def get_resnet_encoder(name: str = "resnet50", pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Ładuje wybrany model ResNet z torchvision i usuwa jego ostatnią warstwę
    klasyfikacyjną (fc), aby działał jako enkoder cech.

    Args:
        name (str): Nazwa modelu ResNet (np. "resnet18", "resnet50").
        pretrained (bool): Czy załadować wagi pre-trenowane na ImageNet.
                           Dla treningu SSL od zera zazwyczaj ustawia się na False.
        **kwargs: Dodatkowe argumenty przekazywane do konstruktora modelu ResNet.

    Returns:
        nn.Module: Model ResNet bez ostatniej warstwy 'fc'.

    Raises:
        ValueError: Jeśli podana nazwa nie jest obsługiwanym ResNetem.
    """
    if name not in SUPPORTED_RESNETS:
        raise ValueError(f"Nieobsługiwany ResNet: {name}. Wybierz z: {SUPPORTED_RESNETS}")

    # Załaduj model ResNet z torchvision
    if pretrained:
        weights = ResNet50_Weights.DEFAULT if name == "resnet50" else "IMAGENET1K_V1"
    else:
        weights = None

    resnet_func = getattr(models, name)
    encoder = resnet_func(weights=weights, **kwargs)
    # Pobierz wymiar cech przed ostatnią warstwą 'fc'
    # W modelach ResNet w torchvision, atrybut 'fc' przechowuje ostatnią warstwę liniową
    try:
        output_feature_dim = encoder.fc.in_features
    except AttributeError:
        # Może się zdarzyć w niektórych niestandardowych wersjach lub jeśli fc już nie istnieje
        print(f"Ostrzeżenie: Nie można automatycznie określić wymiaru wyjściowego dla {name}. Sprawdź architekturę.")
        # Spróbujmy ustalić na podstawie typowej konfiguracji (dla resnet50)
        output_feature_dim = 2048 if name in ["resnet50", "resnet101", "resnet152"] else 512
        print(f"Przyjęto domyślny wymiar cech: {output_feature_dim}")


    # Usuń ostatnią warstwę klasyfikacyjną 'fc'
    # Metoda 1: Zastąp 'fc' warstwą identycznościową
    # encoder.fc = nn.Identity()

    # Metoda 2: Utwórz nowy model sekwencyjny bez 'fc' (bardziej jawne)
    # Lista modułów ResNet bez ostatniej warstwy
    modules = list(encoder.children())[:-1]
    encoder = nn.Sequential(*modules, nn.Flatten())

    # Dodaj atrybut, aby łatwo uzyskać dostęp do wymiaru cech
    # Uwaga: Po nn.Sequential, model może nie mieć atrybutów bezpośrednio
    # Można opakować w dodatkową klasę, jeśli potrzebny jest atrybut 'output_dim'
    # Prostsze rozwiązanie: zwróć wymiar razem z modelem
    # return encoder, output_feature_dim

    # Alternatywnie, jeśli chcemy, aby model miał atrybut .output_dim
    class ResNetWrapper(nn.Module):
        def __init__(self, encoder_seq, dim):
            super().__init__()
            self.encoder = encoder_seq
            self.output_dim = dim

        def forward(self, x):
            # x = self.encoder(x)
            # ResNet zwraca cechy po global average pooling, trzeba je spłaszczyć
            # nn.Flatten jest teraz częścią sekwencji, jeśli użyto metody 2
            # Jeśli nie, dodaj spłaszczenie tutaj lub w modelu nadrzędnym (np. SimCLRNet)
            # Dla bezpieczeństwa, upewnijmy się, że jest spłaszczony:
            return self.encoder(x)

    wrapped_encoder = ResNetWrapper(encoder, output_feature_dim)

    # Zwróć zmodyfikowany enkoder
    # return encoder # Jeśli użyto Metody 1 lub nie potrzebujesz atrybutu .output_dim
    return wrapped_encoder


# Przykład użycia (można umieścić w bloku if __name__ == "__main__":)
if __name__ == '__main__':
    # Pobierz enkoder ResNet50
    encoder = get_resnet_encoder(name="resnet50", pretrained=False)
    print(encoder)
    print(f"Wymiar wyjściowy cech: {encoder.output_dim}")

    # Przykładowe wejście (batch=2, 3 kanały, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224)
    output_features = encoder(dummy_input)
    print(f"Kształt wyjściowych cech: {output_features.shape}") # Oczekiwano: [2, 2048] dla ResNet50