import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(data):
    # Drop irrelevant columns
    data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    # Convert categorical columns
    data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
    data['Geography'] = data['Geography'].astype('category')
    data['HasCrCard'] = data['HasCrCard'].astype('category')
    data['IsActiveMember'] = data['IsActiveMember'].astype('category')
    data['NumOfProducts'] = data['NumOfProducts'].astype('category')
    data['Exited'] = data['Exited'].astype('category')

    # Handle outliers using IQR method
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

    # Split features and target
    X = data.drop('Exited', axis=1)
    y = data['Exited']

    # Check class distribution
    print("Class distribution before SMOTE:")
    print(y.value_counts())

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Handle class imbalance using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled