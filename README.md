# Washington State Real Estate Predictive Analytics Engine

## 📊 Project Overview
This data science project constructs a multivariate linear regression model completely from scratch using `scikit-learn` to forecast residential property values across Washington State. By feeding physical structural characteristics into a predictive pipeline, the model estimates market pricing variations while strictly satisfying classical statistical assumptions.

---

## 🛠️ Key Technical Implementations

*   **Target Optimization**: Identified severe right-skewness within the raw real estate pricing data. Implemented a natural logarithmic transformation (`np.log1p`) to normalize the target distribution, successfully optimizing the pricing skewness score from an imbalanced 0.73 down to a statistically sound -0.42.
*   **Outlier Mitigation**: Calculated statistical dataset boundaries dynamically using the Interquartile Range (IQR) method ($Q3 - Q1$) to drop extreme pricing anomalies without introducing data leakage.
*   **Vectorized Feature Engineering**: Detected a data logging error where unrenovated homes default to a year value of 0. Applied vectorized NumPy logic (`np.where`) to engineer a clean `house_age` predictor relative to the market timeline, reducing structural noise.
*   **Data Leakage Prevention**: Deliberately stripped tracking keys, string text fields, and derivative pricing variables (such as `price_per_sqft`) from the feature matrix $X$ to maintain full predictive integrity.

---

## 📈 Model Performance & Statistical Evaluation

The baseline linear regression engine was trained on an 80% data partition and validated against an unseen 20% testing matrix. 

*   **R² Score (Coefficient of Determination)**: `0.3544` — The engineered physical features successfully account for over 35% of the total variance observed in regional real estate pricing.
*   **RMSE (Root Mean Squared Error)**: `$168,892.58` — The average real-world dollar margin of error for the model's asset valuations.

### 📉 Residual Diagnostics
To ensure model stability, a residual error analysis ($y - \hat{y}$) was conducted. The prediction errors are evenly and randomly distributed above and below the horizontal zero baseline across all pricing tiers. This homoscedastic behavior visually proves that the model's linear assumptions are valid and stable.

---

## 🧰 Tech Stack Used
*   **Data Manipulation**: Python, Pandas, NumPy
*   **Machine Learning**: Scikit-Learn (`LinearRegression`, `train_test_split`)
*   **Data Visualization**: Matplotlib, Seaborn
*   **Environment**: Google Colab Notebook
