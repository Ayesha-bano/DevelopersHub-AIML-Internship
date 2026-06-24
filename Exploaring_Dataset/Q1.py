import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset (built into seaborn)
df = sns.load_dataset('iris')

# Check shape (rows, columns)
print("Shape of dataset:", df.shape)

# Check column names
print("\nColumn names:", df.columns.tolist())

# Print first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Info: data types, non-null counts
print("Dataset Info:")
df.info()

# Describe: mean, std, min, max, percentiles for numeric columns
print("\nDescriptive Statistics:")
print(df.describe())

# Scatter plot - relationships between features
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

# Histogram
df.hist(figsize=(10, 8), bins=20)
plt.suptitle('Histograms of Iris Features')
plt.tight_layout()
plt.show()

# Box plots - identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('Box Plots of Iris Features')
plt.show()