import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# 1) LOAD DATA
DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
raw = pd.read_csv(DATA_PATH)

# 2) DEFINE COLUMNS
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
label_col = "stroke"
num_cols = raw.select_dtypes(include=[np.number]).columns.drop(label_col)

# 3) FILL MISSING VALUES
# Numeric: fill with mean
raw[num_cols] = raw[num_cols].fillna(raw[num_cols].mean())
# Categorical: fill with mode
for c in cat_cols:
    raw[c] = raw[c].fillna(raw[c].mode()[0])

# 4) ENCODE CATEGORICALS
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
raw[cat_cols] = encoder.fit_transform(raw[cat_cols])

# 5) STRATIFIED SPLIT
train_df, test_df = train_test_split(
    raw, test_size=0.2, random_state=42, stratify=raw[label_col]
)
# Further split train into train/validation
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df[label_col]
)

# export raw + splits to CSV
raw.to_csv("raw_df.csv", index=False)
train_df.to_csv("train_df.csv", index=False)
val_df.to_csv("val_df.csv", index=False)
test_df.to_csv("test_df.csv", index=False)
# 6) SCALE + PCA
scaler = StandardScaler().fit(train_df[num_cols])
Xtr_scaled = scaler.transform(train_df[num_cols])
Xval_scaled = scaler.transform(val_df[num_cols])
Xte_scaled = scaler.transform(test_df[num_cols])

pca = PCA(n_components=0.90).fit(Xtr_scaled)
Xtr = pca.transform(Xtr_scaled)
Xval = pca.transform(Xval_scaled)
Xte = pca.transform(Xte_scaled)

ytr = train_df[label_col].values
yval = val_df[label_col].values
yte = test_df[label_col].values

# 7) TORCH DATASET & LOADER
# Convert to torch tensors (no channel dim for fully connected)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1).to(device)
Xval_t = torch.tensor(Xval, dtype=torch.float32).to(device)
yval_t = torch.tensor(yval, dtype=torch.float32).unsqueeze(1).to(device)
Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)
yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1).to(device)

train_ds = TensorDataset(Xtr_t, ytr_t)
val_ds = TensorDataset(Xval_t, yval_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


# 8) MODEL DEFINITION: MLP
class StrokeMLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


n_feats = Xtr.shape[1]
model = StrokeMLP(n_feats).to(device)

# 9) LOSS, OPTIMIZER, SCHEDULER
# Compute class weights
pos = (ytr == 1).sum()
neg = (ytr == 0).sum()
pos_weight = torch.tensor(neg / pos).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

# 10) TRAIN & VALIDATE
n_epochs = 20
best_val_auc = 0.0
for epoch in range(1, n_epochs + 1):
    # Training
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_logits = []
    val_targets = []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            val_logits.append(logits.cpu().numpy())
            val_targets.append(yb.cpu().numpy())
    val_logits = np.vstack(val_logits)
    val_targets = np.vstack(val_targets)
    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
    val_auc = roc_auc_score(val_targets, val_probs)
    scheduler.step(val_auc)

    print(
        f"Epoch {epoch}/{n_epochs}  TrainLoss: {np.mean(train_losses):.4f}  Val AUC: {val_auc:.4f}"
    )

    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_stroke_mlp.pth")

# 11) FINAL EVALUATION ON TEST SET
model.load_state_dict(torch.load("best_stroke_mlp.pth"))
model.eval()
with torch.no_grad():
    logits = model(Xte_t)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

print("Test Accuracy:", accuracy_score(yte, preds))
print(classification_report(yte, preds, zero_division=0))
print("Test ROC AUC:", roc_auc_score(yte, probs))
precision, recall, _ = precision_recall_curve(yte, probs)
print("Test PR AUC:", auc(recall, precision))
