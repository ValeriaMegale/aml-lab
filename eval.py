import torch
import torch.nn as nn
import wandb
import os

# Importa le tue classi definite nelle altre cartelle
from models.customnet import CustomNet
from dataset.tinyimagenet import get_dataloaders

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    print("Inizio validazione...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Results - Loss: {avg_val_loss:.6f} Acc: {val_accuracy:.2f}%')

    # Logga su wandb solo se c'è una run attiva
    if wandb.run is not None:
        wandb.log({"val_loss": avg_val_loss, "val_accuracy": val_accuracy})

    return val_accuracy


if __name__ == "__main__":
    # 1. Configurazione Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Opzionale: Inizializza Wandb se vuoi tracciare anche questa esecuzione singola
    # wandb.init(project="mldl-lab3-tinyimagenet", job_type="evaluation")

    # 3. Carica solo il Dataloader di Validazione
    # Nota: get_dataloaders restituisce (train, val), usiamo _ per ignorare train
    print("Caricamento validation loader...")
    batch_size = 32
    _, val_loader = get_dataloaders(batch_size=batch_size)

    # 4. Inizializza il Modello e la Loss
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()

    # 5. (IMPORTANTE) Carica i pesi se esistono
    # Senza questo, valuti un modello casuale con accuratezza ~0.5%
    checkpoint_path = "checkpoints/model_best.pth"  # O il nome che darai al tuo file salvato
    if os.path.exists(checkpoint_path):
        print(f"Caricamento pesi da {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("⚠️ ATTENZIONE: Nessun checkpoint trovato! Sto valutando un modello non addestrato.")

    # 6. Esegui la validazione
    validate(model, val_loader, criterion, device)