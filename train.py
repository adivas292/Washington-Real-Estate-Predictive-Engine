from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def run_regression_pipeline(X, y):
    """
    Partitions datasets 80/20, fits a multivariate regression engine,
    and runs full statistical diagnostic evaluations.
    """
    # 80/20 Partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Fit Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on log scale
    y_pred_log = model.predict(X_test)
    
    # Invert log data back to standard USD currencies for real-world business tracking
    y_test_usd = np.expm1(y_test)
    y_pred_usd = np.expm1(y_pred_log)
    
    # Statistical Metrics
    r2 = r2_score(y_test, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test_usd, y_pred_usd))
    
    return model, r2, rmse, y_test_usd, y_pred_usd
