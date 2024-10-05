# Steel Plate Defect Prediction

## Overview

This project aims to predict different defects in steel plates using machine learning techniques. The dataset contains various features related to the steel plates, and the goal is to build models that can accurately classify the presence of defects.

## Project Structure

1. **Exploratory Data Analysis (EDA)**:

   - In this phase, the data was thoroughly explored, outliers were identified and removed, and advanced topics such as Variance Inflation Factor (VIF) were discussed to handle multicollinearity issues.
   - EDA was crucial for understanding the characteristics of the dataset and preparing it for model building.
2. **Pipelines**:

   - Pipelines were generated based on the insights gained from EDA. These pipelines helped streamline data preprocessing and feature engineering tasks for subsequent model building.
3. **Machine Learning Model Building**:

   - **Neural Network Approach**: A neural network was trained using both categorical and continuous data. This approach allowed for capturing complex relationships within the data.
   - **XGBoost, LightGBM, and Voting Classifier Approach**: Additionally, XGBoost, LightGBM, and a voting classifier were implemented and compared for their performance. Hyperparameters for XGBoost were tuned using advanced techniques.

## Results and Conclusion

After thorough experimentation, it was found that the neural network outperformed the ensemble methods and gradient boosting models. The neural network exhibited higher accuracy and provided more control over the model's structure. This suggests that for this particular task, the neural network approach is more effective in predicting defects in steel plates.
