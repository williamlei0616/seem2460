import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# Drop the 'id' column
data = data.drop('id', axis=1)

# Handle categorical features by converting them to numeric
# First, check for categorical columns
numerical_data = data.select_dtypes(include=['float64', 'int64'])
categorical_data = data.select_dtypes(include=['object', 'category'])

# If there are categorical columns, you might want to use one-hot encoding
# or other encoding methods before creating the correlation matrix
if not categorical_data.empty:
    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(data)
    # Calculate correlation matrix
    correlation_matrix = data_encoded.corr()
else:
    # Calculate correlation matrix
    correlation_matrix = numerical_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

print("Correlation matrix created successfully!")