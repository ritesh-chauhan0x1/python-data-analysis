data = pd.DataFrame({'x': range(10), 'y': [i**2 for i in range(10)]})

# Python Data Analysis Project - Enhanced Version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_sample_csv(filename):
	np.random.seed(42)
	df = pd.DataFrame({
		'id': range(1, 101),
		'age': np.random.randint(18, 70, 100),
		'income': np.random.normal(50000, 15000, 100).astype(int),
		'score': np.random.uniform(0, 100, 100),
		'group': np.random.choice(['A', 'B', 'C'], 100)
	})
	df.loc[np.random.choice(df.index, 10, replace=False), 'income'] = np.nan  # introduce NaNs
	df.to_csv(filename, index=False)
	print(f"Sample data written to {filename}")

def load_data(filename):
	print(f"Loading data from {filename}...")
	return pd.read_csv(filename)

def clean_data(df):
	print("Cleaning data...")
	print(f"Missing values before cleaning:\n{df.isnull().sum()}")
	df['income'].fillna(df['income'].median(), inplace=True)
	print(f"Missing values after cleaning:\n{df.isnull().sum()}")
	return df

def analyze_data(df):
	print("\nBasic statistics:")
	print(df.describe())
	print("\nGroup counts:")
	print(df['group'].value_counts())

def visualize_data(df):
	print("Generating plots...")
	os.makedirs('plots', exist_ok=True)
	# Histogram
	plt.figure(figsize=(8,4))
	sns.histplot(df['age'], bins=15, kde=True)
	plt.title('Age Distribution')
	plt.savefig('plots/age_distribution.png')
	plt.close()

	# Boxplot
	plt.figure(figsize=(8,4))
	sns.boxplot(x='group', y='income', data=df)
	plt.title('Income by Group')
	plt.savefig('plots/income_by_group.png')
	plt.close()

	# Scatter plot
	plt.figure(figsize=(8,4))
	sns.scatterplot(x='age', y='score', hue='group', data=df)
	plt.title('Score vs Age by Group')
	plt.savefig('plots/score_vs_age.png')
	plt.close()

	# Correlation heatmap
	plt.figure(figsize=(6,5))
	sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
	plt.title('Correlation Matrix')
	plt.savefig('plots/correlation_matrix.png')
	plt.close()

def report(df):
	print("\n--- Report ---")
	print(f"Total records: {len(df)}")
	print(f"Average income: {df['income'].mean():.2f}")
	print(f"Average score: {df['score'].mean():.2f}")
	print(f"Group breakdown:\n{df['group'].value_counts()}")

def main():
	filename = 'sample_data.csv'
	if not os.path.exists(filename):
		generate_sample_csv(filename)
	df = load_data(filename)
	df = clean_data(df)
	analyze_data(df)
	visualize_data(df)
	report(df)
	print("\nPlots saved in the 'plots' directory.")

if __name__ == "__main__":
	main()
