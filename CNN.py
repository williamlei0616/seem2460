import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim


class TabularDataset(Dataset):
    def __init__(self, csv_file, scaler=None, fit_scaler=False):
        df = pd.read_csv(csv_file)
        # drop unused columns
        X = df.drop(columns=["id", "stroke"]).values.astype(np.float32)
        y = df["stroke"].values.astype(np.float32)
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # add channel dim for Conv1d: (C=1, L=features)
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
        y = torch.tensor(self.y[idx])
        return x, y


class CNN1D(nn.Module):
    def __init__(self, in_channels=1, num_features=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv(x)  # shape: (batch, 256, 1)
        x = self.fc(x)  # shape: (batch, 1)
        return x.squeeze(1)  # shape: (batch,)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            ys.append(y.numpy().astype(int))
            preds.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load datasets
    train_ds = TabularDataset("train_df.csv", fit_scaler=True)
    val_ds = TabularDataset("val_df.csv", scaler=train_ds.scaler, fit_scaler=False)
    test_ds = TabularDataset("test_df.csv", scaler=train_ds.scaler, fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = CNN1D(in_channels=1, num_features=train_ds.X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_f1 = 0
    for epoch in range(1, 21):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "best_model.pth")
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} F1: {val_metrics['f1']:.4f}"
        )

    # load best model and evaluate on test set
    model.load_state_dict(torch.load("best_model.pth"))
    test_metrics = evaluate(model, test_loader, device)
    print(
        "Test Results:",
        f"Acc: {test_metrics['acc']:.4f}",
        f"Precision: {test_metrics['prec']:.4f}",
        f"Recall: {test_metrics['rec']:.4f}",
        f"F1: {test_metrics['f1']:.4f}",
    )


if __name__ == "__main__":
    main()
