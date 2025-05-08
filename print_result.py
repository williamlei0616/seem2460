import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the saved model
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')

# Load test data
test_df = pd.read_csv('test_df.csv')
test_df = test_df.drop(columns=['id'])
X_test = test_df.drop(columns=['stroke'])
y_test = test_df['stroke']

# Make predictions
y_pred = model.predict(X_test)

# Compute metrics without averaging (per class)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision per class:", precision)
print("Recall per class:", recall)
print("F1 Score per class:", f1)

# Plot feature importance as a bar graph and save the image
feature_importance = model.feature_importances_
indices = feature_importance.argsort()[::-1]
feature_names = X_test.columns

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=90)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()