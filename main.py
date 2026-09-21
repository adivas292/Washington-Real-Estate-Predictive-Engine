from src.preprocessing import preprocess_housing_data
from src.train import run_regression_pipeline

if __name__ == "__main__":
    print("🚀 Initializing Production Real Estate Analytics Pipeline...")
    
    # Path to data asset
    data_path = "data/housing_data.csv"
    
    # Run Modular Pipelines
    X, y = preprocess_housing_data(data_path)
    model, r2, rmse, actual_usd, pred_usd = run_regression_pipeline(X, y)
    
    # Print Diagnostics to Terminal
    print("\n--- MODEL PERFORMANCE DIAGNOSTICS ---")
    print(f"✅ Multivariate Feature Shape: {X.shape}")
    print(f"✅ Coefficient of Determination (R²): {r2:.4f}")
    print(f"✅ Real-World Valuation Margin of Error (RMSE): ${rmse:,.2f}\n")
