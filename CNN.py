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
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
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
raw[num_cols] = raw[num_cols].fillna(raw[num_cols].mean())
for c in cat_cols:
    raw[c] = raw[c].fillna(raw[c].mode()[0])

# 4) ENCODE CATEGORICALS
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
raw[cat_cols] = encoder.fit_transform(raw[cat_cols])

# 5) STRATIFIED SPLIT (before SMOTE)
train_df, test_df = train_test_split(
    raw, test_size=0.2, random_state=8964, stratify=raw[label_col]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=8964, stratify=train_df[label_col]
)

# 6) DOWNSAMPLE MAJORITY CLASS THEN APPLY SMOTE
majority = train_df[train_df[label_col] == 0]
minority = train_df[train_df[label_col] == 1]

majority_down = resample(
    majority, replace=False, n_samples=len(minority), random_state=42
)
train_downsampled = pd.concat([majority_down, minority])

X_train = train_downsampled.drop(columns=[label_col])
y_train = train_downsampled[label_col]

smote = SMOTE(random_state=42)
Xtr_sm, ytr_sm = smote.fit_resample(X_train, y_train)

# 7) SCALE + PCA
scaler = StandardScaler().fit(Xtr_sm[num_cols])
Xtr_scaled = scaler.transform(Xtr_sm[num_cols])
Xval_scaled = scaler.transform(val_df[num_cols])
Xte_scaled = scaler.transform(test_df[num_cols])

pca = PCA(n_components=0.99).fit(Xtr_scaled)
Xtr = pca.transform(Xtr_scaled)
Xval = pca.transform(Xval_scaled)
Xte = pca.transform(Xte_scaled)

ytr = ytr_sm.values
yval = val_df[label_col].values
yte = test_df[label_col].values

# 8) TORCH DATASET & LOADER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1).to(device)
Xval_t = torch.tensor(Xval, dtype=torch.float32).to(device)
yval_t = torch.tensor(yval, dtype=torch.float32).unsqueeze(1).to(device)
Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)
yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1).to(device)

train_ds = TensorDataset(Xtr_t, ytr_t)
val_ds = TensorDataset(Xval_t, yval_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)


# 9) MODEL DEFINITION: MLP
class StrokeMLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


n_feats = Xtr.shape[1]
model = StrokeMLP(n_feats).to(device)

# 10) LOSS, OPTIMIZER, SCHEDULER
pos = (ytr == 1).sum()
neg = (ytr == 0).sum()
pos_weight = torch.tensor(neg / pos).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

# 11) TRAINING WITH EARLY STOPPING
n_epochs = 30
patience = 3
best_val_auc = 0.0
epochs_no_improve = 0

for epoch in range(1, n_epochs + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

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

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_stroke_mlp.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# 12) FINAL EVALUATION ON TEST SET
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

# 13) CLASS-SPECIFIC ACCURACY
true_neg = np.sum((yte == 0) & (preds.flatten() == 0))
true_pos = np.sum((yte == 1) & (preds.flatten() == 1))
total_neg = np.sum(yte == 0)
total_pos = np.sum(yte == 1)

print("Negative class accuracy:", true_neg / total_neg)
print("Positive class accuracy:", true_pos / total_pos)
