import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('all_india_dataset_final.csv')

# Display the first few rows of the dataset
print(df.head())

# Check the dataset size
df.shape

# Check for null values
df.isnull().sum()

# Check for missing values
df.isna().sum()

# Drop the first column as it has no meaning
df = df.drop(df.columns[0], axis=1)
df.head()

# Drop the rows with missing values
df = df.dropna()

# Outlier detection and removal
z_score = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df = df[(z_score < 3).all(axis=1)]

# Standardization of numerical columns
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# One-hot encoding of categorical columns
encoder = OneHotEncoder()
categorical_cols = df.select_dtypes(include=[object]).columns
encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]).toarray(), columns=encoder.get_feature_names(categorical_cols))
df = pd.concat([df, encoded_cols], axis=1)
df = df.drop(categorical_cols, axis=1, inplace=True)

# Split the dataset into training and testing sets, 80-20 split
X = df.drop('target', axis=1) # change to actual target = crop yield???
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the cleaned dataset
df.to_csv('cleaned_all_india_dataset.csv', index=False)
