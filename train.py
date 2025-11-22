import torch
from torch import nn
import torch.optim as optim

# Importa dai tuoi file .py
from models.customnet import CustomNet
from dataset.tinyimagenet import get_dataloaders
from eval import validate

import wandb

# --- Funzione di Training (da Cella 9) ---
def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device) # Sposta su device

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    # Logica Wandb (la aggiungiamo tra poco)
    # wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "epoch": epoch})


# --- Main execution (da Cella 11) ---
if __name__ == "__main__":
    
    # Imposta il device (es. per Colab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    wandb.init(project="mldl-lab3-tinyimagenet")
    
    # Iperparametri
    batch_size = 32
    lr = 0.001
    momentum = 0.9
    num_epochs = 10
    
    wandb.config.update({
         "batch_size": batch_size,
         "learning_rate": lr,
         "momentum": momentum,
         "epochs": num_epochs
     })

    # Carica i Dati
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    
    # Inizializza Modello, Criterio, Ottimizzatore
    model = CustomNet().to(device) # Sposta il modello sul device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_acc = 0

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer, device)
        val_accuracy = validate(model, val_loader, criterion, device)

        best_acc = max(best_acc, val_accuracy)

    print(f'Best validation accuracy: {best_acc:.2f}%')
    
    checkpoint_path = "checkpoints/model_best.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Modello salvato in {checkpoint_path}")