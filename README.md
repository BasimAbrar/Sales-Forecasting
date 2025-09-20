# Sales-Forecasting
Walmart sales forecasting using historical data. Built time-based features, applied Linear Regression and XGBoost. XGBoost achieved R²=0.97 and RMSE=3296, capturing both trend and seasonality.

## Dataset  
- **Source:** [Walmart Sales Forecast Dataset – Kaggle]  
- Includes historical weekly sales for different stores and departments, along with additional features such as holidays and markdowns.  

## Steps Performed  
1. Preprocessed dataset and created time-based features (day, month, lag values).  
2. Built regression models:  
   - Linear Regression  
   - XGBoost Regressor  
3. Evaluated models using **R²** and **RMSE**.  
4. Visualized actual vs. predicted sales over time.  

## Results  
- **Linear Regression**: R² = 0.9135, RMSE = 5601.89  
- **XGBoost**: R² = 0.9701, RMSE = 3296.10  

**Observation:**  
- Linear Regression captured the general trend but missed seasonal variations.  
- XGBoost performed better, capturing both trend and seasonality.  

## Tools & Libraries  
- Python  
- Pandas  
- Scikit-learn  
- XGBoost  
- Matplotlib  

## Conclusion  
XGBoost Regressor outperformed Linear Regression, achieving higher accuracy and better handling of sales seasonality. It is the most reliable model for Walmart sales forecasting.  
