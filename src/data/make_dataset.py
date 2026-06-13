import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_and_preprocess_adult(test_size=0.3, random_state=42):
    """
    Fetches the Adult Census Income dataset from OpenML,
    preprocesses it, and returns train/test splits.
    """
    print("Fetching Adult dataset from OpenML...")
    # Fetch adult dataset (ID: 1590)
    data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
    df = data.frame
    
    # Drop rows with missing values for simplicity
    df = df.dropna()
    
    # Define our target: income >50K
    # The 'class' column contains '<=50K' and '>50K'
    df['income'] = (df['class'] == '>50K').astype(int)
    df = df.drop(columns=['class'])
    
    # Encode protected attributes
    # 'sex' is typically 'Male' / 'Female'. Encode as 1 / 0
    df['sex'] = (df['sex'] == 'Male').astype(int)
    
    # 'race' - simplify to 'White' vs 'Non-White' for binary bias analysis
    df['race'] = (df['race'] == 'White').astype(int)
    
    # Select a subset of features for the causal model
    features = [
        'age', 'workclass', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 
        'race', 'sex', 'capital-gain', 'capital-loss', 
        'hours-per-week', 'native-country'
    ]
    
    # Encode categorical variables
    categorical_cols = df[features].select_dtypes(include=['category', 'object']).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    # Split the dataset
    X = df[features]
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, encoders, df

if __name__ == "__main__":
    load_and_preprocess_adult()
