import numpy as np
import pandas as pd

def preprocess_housing_data(file_path):
    """
    Cleans raw data anomalies, clips IQR outliers, normalizes the target variable,
    and isolates the full multi-variable feature matrix.
    """
    # Load raw file
    df = pd.read_csv(file_path)
    
    # Phase 1: Clean impossible structural logging errors
    df = df[df['bathrooms'] > 0]
    df = df[df['price'] > 0]
    
    # Phase 2: Dynamic IQR Outlier Mitigation
    Q1 = 3.261000e+05
    Q3 = 6.575000e+05
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    
    # Target Transformation to resolve right-skewness
    df['log_price'] = np.log1p(df['price'])
    
    # Vectorized Feature Engineering: Resolve missing renovation logs
    df['effective_year'] = np.where(df['yr_renovated'] > 0, df['yr_renovated'], df['yr_built'])
    df['house_age'] = 2015 - df['effective_year']
    
    # Data Leakage Prevention: Drop targets, strings, and derivative trackers
    # Note: We KEEP the numerical physical features (bedrooms, bathrooms, sqft_living, floors, water-front, view, condition)
    X = df.drop(axis=1, columns=[
        'price', 'log_price', 'date', 'street', 'city', 'statezip', 
        'yr_built', 'yr_renovated', 'effective_year'
    ])
    
    # If 'price_per_sqft' or other leaky columns exist in the raw set, drop them safely
    if 'price_per_sqft' in X.columns:
        X = X.drop(columns=['price_per_sqft'], axis=1)
        
    y = df['log_price']
    
    return X, y
