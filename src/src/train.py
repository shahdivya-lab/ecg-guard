import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import ECGGuard


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for ecg, labels in loader:
        ecg, labels = ecg.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(ecg)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for ecg, labels in loader:
            ecg, labels = ecg.to(device), labels.to(device)
            output = model(ecg)
            loss = criterion(output, labels)
            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = ECGGuard(num_classes=5, confidence_threshold=0.75).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Replace with your actual PTB-XL dataset loader
    # dataset = PTBXLDataset("path/to/ptbxl")
    # train_set, val_set = random_split(dataset, [0.8, 0.2])
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    # val_loader   = DataLoader(val_set,   batch_size=32)

    print("Model ready. Plug in PTB-XL dataset to begin training.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
```

4. Commit changes.

---

Your repo now has this structure:
```
ecg-guard/
├── README.md
├── requirements.txt
└── src/
    ├── model.py
    └── train.py
