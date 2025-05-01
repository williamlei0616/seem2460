import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Train SVM with class_weight balanced
svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
svm.fit(X_train_scaled, y_train)

# Validation predictions
val_preds = svm.predict(X_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds))

# Test predictions
test_preds = svm.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds))