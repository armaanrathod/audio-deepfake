import torch, torch.nn as nn
import numpy as np, pandas as pd, os
from torch.utils.data import DataLoader, Dataset
from model import DeepfakeCNN

class SpectrogramDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec = np.load(row['spec_path'])             # (128, 126)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, 128, 126)
        label = torch.tensor(float(row['label']), dtype=torch.float32)
        return spec, label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = SpectrogramDataset("labels_balanced.csv")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    model = DeepfakeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(10):
        total, correct, running_loss = 0, 0, 0
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(specs).squeeze(1)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += ((preds > 0.5) == labels.bool()).sum().item()
            total += len(labels)
        print(f"Epoch {epoch+1:02d} | Loss: {running_loss/len(loader):.4f} | Acc: {correct/total*100:.1f}%")

    torch.save(model.state_dict(), "model.pth")
    print("Saved → model.pth")

if __name__ == "__main__": train()