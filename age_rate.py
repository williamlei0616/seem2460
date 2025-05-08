import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# Group by age and calculate stroke rate
stroke_by_age = data.groupby('age')['stroke'].mean().reset_index()
stroke_by_age['stroke_rate'] = stroke_by_age['stroke'] * 100  # Convert to percentage

# Plot the stroke rate by age
plt.figure(figsize=(12, 6))
sns.lineplot(x='age', y='stroke_rate', data=stroke_by_age, marker='o')
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Stroke Rate by Age')
plt.xlabel('Age')
plt.ylabel('Stroke Rate (%)')
plt.tight_layout()

# Save the plot
plt.savefig('stroke_rate_by_age.png')
plt.show()

# Print summary statistics
print("Stroke Rate by Age Summary:")
print(stroke_by_age.sort_values(by='stroke_rate', ascending=False).head(10))