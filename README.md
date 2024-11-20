# Breast-Cancer-Detection-ML

This project aims to classify breast cancer cases as malignant or benign using the Breast Cancer Wisconsin Diagnosis dataset. The implementation includes comprehensive steps to ensure accurate predictions and insightful data analysis. Below is a detailed overview of the methods, techniques, and tools used in the project.

Key Highlights
Exploratory Data Analysis (EDA):

Performed detailed EDA to understand the data distribution and relationships.
Created visualizations like correlation heatmaps, distribution plots, and target variable analysis.
Observed multicollinearity among features, aiding in dimensionality reduction.
Dimensionality Reduction with PCA:

Identified and resolved multicollinearity issues by using Principal Component Analysis (PCA).
Reduced the dataset to components explaining ~95% of the variance, optimizing the feature space for modeling.
Model Building - Logistic Regression:

Developed a Logistic Regression model for classification.
Split the dataset into training and testing sets using an 80:20 ratio.
Trained the model on principal components to ensure high performance and reduced computational overhead.
Evaluation Metrics:

Evaluated the model’s performance using:
Confusion Matrix: Visualized true positives, true negatives, false positives, and false negatives.
Classification Report: Displayed precision, recall, and F1-score for both classes.
ROC Curve and AUC: Assessed model performance across thresholds, achieving a high AUC score.
Predictive Capabilities:

Designed a pipeline to handle new data, including preprocessing (scaling and PCA) and classification.
Saved and loaded the trained model, PCA pipeline, and scaler for reuse.
Visualizing Results:

Presented key insights and results through visualizations like the ROC Curve, Confusion Matrix, and Cumulative Explained Variance Plot.
Created a final summary graph combining metrics like accuracy, precision, recall, and F1-score for better interpretability.
Project Workflow
Data Loading and Preprocessing:

Cleaned the dataset by removing unnecessary columns and mapping the target variable to binary (0 for benign, 1 for malignant).
Scaled the data using StandardScaler to ensure uniformity across features.
Feature Selection and Engineering:

Analyzed multicollinearity using Variance Inflation Factor (VIF) to identify redundant features.
Dropped highly collinear features before applying PCA for dimensionality reduction.
Model Training and Testing:

Built a Logistic Regression model on the PCA-transformed data.
Split the dataset into training and testing sets for model evaluation.
Saving and Reloading the Model:

Used joblib to save the trained model, PCA pipeline, and scaler for future use.
Implemented functionality to load the saved model and make predictions on new data.
Performance Analysis:

Evaluated model performance using multiple metrics.
Tuned thresholds to balance precision and recall for optimal classification results.
Technologies and Tools Used
Programming Language: Python
Libraries:
Data Manipulation: pandas, numpy
Visualization: matplotlib, seaborn
Machine Learning: scikit-learn, statsmodels
Model Saving: joblib
Project Outcomes
Accuracy: The model achieved a high classification accuracy of 98% on the test set.
Insights: Identified key features influencing predictions and reduced dimensionality while preserving essential information.
Reusability: The trained model and pipelines can be reused to predict new cases efficiently.
Future Enhancements
Experiment with other classifiers like Random Forest or SVM for comparison.
Implement hyperparameter tuning to further optimize Logistic Regression performance.
Deploy the model as a web application for real-time predictions.
