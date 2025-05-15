import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_values(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_feature_distributions(data):
    data.hist(bins=30, figsize=(15, 10))
    plt.suptitle('Feature Distributions')
    plt.show()

def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 10))
    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_outliers(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    num_columns = len(numeric_columns)
    
    # Determine grid size
    cols = 2  # Number of columns in the grid
    rows = (num_columns + cols - 1) // cols  # Calculate number of rows needed

    plt.figure(figsize=(15, 5 * rows))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x=data[column])
        plt.title(f'Box Plot of {column}')
    plt.tight_layout()
    plt.show()