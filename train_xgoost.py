import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')
val_df = pd.read_csv('val_df.csv')
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])
val_df = val_df.drop(columns=['id'])
X_train = train_df.drop(columns=['stroke'])
y_train = train_df['stroke']
X_test = test_df.drop(columns=['stroke'])
y_test = test_df['stroke']
X_val = val_df.drop(columns=['stroke'])
y_val = val_df['stroke']
kfold = KFold(n_splits = 10, random_state = None, shuffle = False)

for train_index, val_index in kfold.split(train_df):
    train = train_df.iloc[train_index]
    validation = train_df.iloc[val_index]
params = {
          'booster' : ['gbtree'],    
          'n_estimators' : [ 800, 1000],
          'objective' : ['reg:squarederror'],
          'learning_rate': [0.4, 0.5],
          'gamma' : [0.25, 0.5, ],
          'alpha' : [0],     # L1 (Lasso Regression) Regularization Parameter (min = 0)
          'lambda' : [1.25, 1.5], # L2 (Ridge Regression) Regularization Parameter
          'max_depth': [3, 4],
          'min_child_weight': [.6, 0.8],
          'random_state' : [42]
         }
model = xgb.XGBClassifier()
grid_search = GridSearchCV(estimator = model, 
                           param_grid = params, 
                           scoring = 'neg_mean_squared_error',
                           n_jobs = -1, 
                           cv = kfold,  
                           refit = True).fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
# Save the best model
model = xgb.XGBClassifier(**best_params)
model.fit(X_test, y_test)
model.save_model('xgboost_model.json')

# Show performance with visualizations
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, classification_report
import numpy as np
from matplotlib.ticker import PercentFormatter

# Make predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

# Create a figure with multiple subplots
plt.figure(figsize=(20, 15))

# 1. Confusion Matrix
plt.subplot(2, 2, 1)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Stroke', 'Stroke'])
plt.yticks(tick_marks, ['No Stroke', 'Stroke'])
thresh = conf_matrix.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
plt.xlabel('Predicted label')
plt.ylabel('True label')

# 2. ROC Curve
plt.subplot(2, 2, 2)
plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# 3. Precision-Recall Curve
plt.subplot(2, 2, 3)
plt.plot(recall, precision, 'g-')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.ylim([0, 1.05])
plt.xlim([0, 1])

# 4. Feature Importance
plt.subplot(2, 2, 4)
feature_importance = model.feature_importances_
indices = np.argsort(feature_importance)[::-1]
feature_names = X_test.columns
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=90)
plt.title('Feature Importance')
plt.tight_layout()

# Print metrics summary
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the figure
plt.savefig('xgboost_performance.png', dpi=300, bbox_inches='tight')
plt.show()

