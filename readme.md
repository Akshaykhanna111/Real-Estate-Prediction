# House Price Prediction with XGBoost Regression

### Overview
This project applies XGBoost Regression techniques to predict house prices based on the famous Kaggle Dataset "House Prices - Advanced Regression Techniques." The goal is to build a predictive model that accurately estimates the sale price of houses in the dataset.

### Dependencies

- pandas
- scikit-learn
- xgboost
- numpy
- scipy

### Dataset
The dataset used for this analysis is the "House Prices - Advanced Regression Techniques" dataset from Kaggle. It contains information about various features of houses and their sale prices.

### Files

- train.csv: Training data containing features and sale prices of houses.
- test.csv: Test data containing features of houses for which sale prices need to be predicted.
- submission.csv: File containing the predicted sale prices for the test dataset.

### Process

- Data Exploration: Initial exploration of the training data to understand its structure and characteristics.
- Feature Selection: Retained columns that are statistically significant based on T-tests for categorical columns and correlation for numerical columns.
- Model Building: Trained an XGBoost model on the selected features to predict house prices.
- Model Evaluation: Evaluated the model's performance using Mean Squared Error (MSE) on a validation set.
- Hyperparameter Tuning: Performed hyperparameter tuning using GridSearchCV to find the best model parameters.
- Final Model Training: Retrained the model on the entire training dataset (including the validation set) with the best hyperparameters.
- Prediction on Test Data: Applied the final model to predict house prices on the test dataset and saved the predictions in submission.csv.

### Results

The model's performance was evaluated using Mean Squared Error (MSE) on the validation set. Hyperparameter tuning was performed to improve the model's accuracy. The final predictions for the test dataset were saved in submission.csv.

### Challenges

- Dealing with missing values and selecting the most relevant features for the model.
- Balancing model complexity and performance to avoid overfitting.

### Future Work

- Exploring more advanced feature engineering techniques to improve model performance.
- Experimenting with different machine learning algorithms and ensemble methods.
