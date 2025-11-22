import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32, root_dir='tiny-imagenet/tiny-imagenet-200'):
    # --- 1. Trasformazioni per il TRAINING (con Augmentation) ---
    train_transform = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 2. Trasformazioni per la VALIDAZIONE (senza Augmentation) ---
    val_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 3. Carica i dataset ---
    train_dataset = ImageFolder(
        root=f'{root_dir}/train',
        transform=train_transform
    )
    val_dataset = ImageFolder(
        root=f'{root_dir}/val',
        transform=val_transform
    )

    # --- 4. Crea i DataLoader ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print("Dataloaders creati.")
    return train_loader, val_loader