# 🏠 Washington State Real Estate Predictive Engine

## 👋 Project Overview
I built this project to see how well standard physical features (like square footage, bedrooms, and the age of a house) can actually predict home prices in Washington State. 

This project started out as a messy, exploratory analysis in a Google Colab notebook using a dataset I found on Kaggle. After practicing my data science skills there, I used Gemini to help me figure out how to refactor my code blocks, debug my syntax errors, and organize everything into structured, modular Python files (`preprocessing.py`, `train.py`, and `main.py`). 

My goal was to connect what I learned in AP Statistics and my Inspirit AI training to a real-world software pipeline.

---

## 🛠️ My Data Decisions & What I Learned

*   **Fixing Skewed Prices**: When I first looked at the housing data, the prices were heavily right-skewed because of a few multi-million dollar mansions. To stop these from throwing off a basic linear regression model, I applied a natural log transformation (`np.log1p`). This brought my target skewness score down from a heavily skewed 0.73 to a much more balanced -0.42.
*   **Cutting Outliers**: I used the Interquartile Range (IQR) method (\(Q3 - Q1\)) to calculate upper and lower bounds based on my data summary, and manually dropped the extreme pricing anomalies so they wouldn't warp the baseline model.
*   **Fixing Data Errors**: I noticed a weird logging glitch in the raw data where houses that had never been renovated defaulted to a year value of `0`. If I left it that way, it would break my calculations, so I used vectorized NumPy logic (`np.where`) to calculate a clean, realistic `house_age` instead.
*   **Stopping Data Leakage**: I made sure to manually drop fields like dates, street names, cities, and especially `price_per_sqft` from my feature matrix. Keeping `price_per_sqft` in the data would be cheating, since it directly leaks the final price we are trying to predict!

---

## 📈 Testing the Model & Results

I split my data into an 80% training set and a 20% testing set using `scikit-learn` to see how the model would perform on houses it had never seen before. After running the predictions, I used `np.expm1()` to turn the log prices back into normal U.S. Dollars.

*   **R² Score**: `0.3544` — This means my basic physical features successfully account for about 35.4% of the variance in Washington house prices. It's a solid start for a baseline model, but it proves that physical traits alone don't tell the whole story (location and zip codes matter a lot!).
*   **RMSE (Margin of Error)**: `$168,892.58` — On average, my model's price estimates miss the actual real-world value by about \$168k. 

### 📉 Checking My Residuals
I plotted a residual error chart (\(actual - predicted\)) to make sure a linear model was the right choice. The data points are scattered pretty evenly above and below the zero line across the board, which visually proves that the baseline model is statistically stable and my linear assumptions held up.

---

## 🧰 Tech Stack
*   **Languages**: Python
*   **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
*   **Environment**: Google Colab Notebook & Local IDE

---

## 📁 Repository Structure & Journey
*   `Washington_State_Housing_Prices.ipynb`: My original, experimental scratchpad where I downloaded the data, tested my math, and made my first plots to see the patterns. 
*   `preprocessing.py`, `train.py`, `main.py`: The final production-ready modules I built after using Gemini to help teach me how to modularize my notebook code.
