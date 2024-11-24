# Breast-Cancer-Detection-Project-
This is a classification machine learning model for breast cancer detection
Breast Cancer Detection: Project Report
1. Objective
The objective of this project was to build an efficient machine learning classification model to accurately detect breast cancer based on various diagnostic features. The goal was to preprocess the data, identify and handle outliers, select relevant features, and evaluate multiple classification models to find the best-performing one.

2.Dataset
Dataset Description:

The dataset contains diagnostic measurements of breast cancer tumors, such as mean radius, mean area, and texture.
It includes a target variable indicating whether the tumor is benign (0) or malignant (1).
Dataset Summary:

Shape: Rows: <number of rows>; Columns: <number of columns>
Features: Numerical diagnostic features.
Target Variable: Binary classification (0 = Benign, 1 = Malignant).
Data Quality:

No missing values.
No duplicate entries.

3.Exploratory Data Analysis (EDA)
Descriptive Statistics:

Basic details were fetched using info(), describe(), and shape.
Features were analyzed for statistical properties like mean, standard deviation, min, and max values.
Outlier Detection and Treatment:

Boxplots were drawn for individual features and collectively to visualize outliers.
Outliers were identified using the IQR (Interquartile Range) method.
Capping was applied to fix outliers for all features except the target variable.
Post-capping, boxplots were re-drawn to verify the changes.

Skewness:

Skewness values were checked and found to be within acceptable ranges.
Histograms were plotted to visualize feature distributions.
Correlation Analysis:

A correlation heatmap was created to understand the relationships between features.
Scatterplots were added (e.g., mean radius vs. mean area) for specific feature pairs.

4. Data Preprocessing
Feature Selection:

A correlation matrix was used to identify and select the most relevant features.
Feature Scaling:

Applied MinMax Scaler to normalize features and bring them to a common scale.
Train-Test Split:

The data was split into training and testing sets to evaluate model performance.

5. Model Building and Evaluation
Five machine learning models were trained and evaluated:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)
k-Nearest Neighbors (k-NN)
Performance Comparison:

Logistic Regression: Accuracy = 0.9825
Decision Tree Classifier: Accuracy = 0.9211
Random Forest Classifier: Accuracy = 0.9561
SVM: Accuracy = 0.9825
k-NN: Accuracy = 0.9474


6. Model Optimization
Hyperparameter Tuning:
The Logistic Regression model was optimized using hyperparameter tuning techniques.
Post-tuning, the model retained its high accuracy and robust performance metrics.

7. Model Deployment
The tuned Logistic Regression model was saved for future use, ensuring it is ready for deployment in real-world scenarios.

8. Conclusion
Key Findings:

Logistic Regression emerged as the best model with an accuracy of 98.25%.
Outlier treatment and feature scaling significantly improved model performance.
EDA and correlation analysis provided valuable insights into feature relationships.
Impact:

The project demonstrates how machine learning can effectively detect breast cancer, aiding in early diagnosis and better treatment planning.

9. Future Work
Experiment with advanced ensemble models like XGBoost or LightGBM.
Perform further feature engineering, including interaction terms or polynomial features.
Incorporate cross-validation for more robust performance evaluation.

