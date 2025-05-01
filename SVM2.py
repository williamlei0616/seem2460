import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Add this import
from imblearn.over_sampling import SMOTE

# Load data
train_df = pd.read_csv("train_df.csv")
val_df = pd.read_csv("val_df.csv")
test_df = pd.read_csv("test_df.csv")

X_train = train_df.drop(columns=["stroke"])
y_train = train_df["stroke"]

X_val = val_df.drop(columns=["stroke"])
y_val = val_df["stroke"]

X_test = test_df.drop(columns=["stroke"])
y_test = test_df["stroke"]

# Show class distribution
print("Train class distribution:\n", y_train.value_counts())
print("Validation class distribution:\n", y_val.value_counts())
print("Test class distribution:\n", y_test.value_counts())

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("Resampled train class distribution:\n", pd.Series(y_train_res).value_counts())

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']
}
grid = GridSearchCV(
    SVC(probability=True, random_state=42, class_weight='balanced'),
    param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1
)
grid.fit(X_train_res, y_train_res)
print("Best parameters found by GridSearchCV:", grid.best_params_)

# Use the best estimator for predictions
best_svm = grid.best_estimator_

# Validation predictions
val_preds = best_svm.predict(X_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds))

# Test predictions
test_preds = best_svm.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds))