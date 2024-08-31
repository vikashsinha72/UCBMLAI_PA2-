# UCBMLAI_PA3
Practical Application III for Comparing Classifiers

## Overview
In this practical application assignment, the goal is to compare the performance of various classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines). The dataset used is related to bank product marketing over the telephone, which helps in predicting whether a client will subscribe to a term deposit. The dataset can be found at [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).

The CRISP-DM methodology has been utilized to understand and analyze the data for comparing the performance of each classifier.

## Understanding the Data
- **Data Source**: "data/bank-additional/bank-additional-full.csv"
- **Total Columns**: 21
- **Total Rows**: 41188
- Data was clean anf no null data found.
- Enough data to train all tried models. 
- The data includes features such as age, job, marital status, education, default status, balance, and more, which are used to predict the target variable (whether the client subscribed to a term deposit).

## Understand the Business Problem
- **Objective**: The business goal is to predict whether a client will subscribe to a term deposit based on the results of telemarketing campaigns.
- **Use Case**: This prediction will help the bank target potential customers more effectively, optimizing marketing strategies and improving conversion rates.

## Data Understanding and Preparation
### Explore the Dataset
- Descriptive statistics and visualizations are used to understand the features and their relationships.
- **Encoding Categorical Variables**: Categorical features are converted into numerical values using techniques label encoding for categorical variables.
- **Feature Scaling**: Feature scaling (e.g., standardization) is applied, particularly for algorithms like SVM and K-Nearest Neighbors.

### Splitting the Data
- The dataset is split into training and test sets (e.g., 80% train, 20% test) to evaluate model performance.

### A Simple Model
Used Logistic Regression to build a basic model on data.
Accuracy: 0.91
Confusion Matrix:
[[7108  195]
 [ 542  393]]
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.97      0.95      7303
           1       0.67      0.42      0.52       935

    accuracy                           0.91      8238
   macro avg       0.80      0.70      0.73      8238
weighted avg       0.90      0.91      0.90      8238

The Logistic Regression model performs well overall, particularly in identifying clients who did not subscribe to a term deposit. However, it has a lower performance in identifying clients who did subscribe, as indicated by the lower precision, recall, and F1-score for class 1. Improving the model's ability to predict the minority class (subscribers) will be a key focus in future model iterations, potentially through techniques like class balancing, feature engineering, or hyperparameter tuning.


## Modeling with Different Classifiers
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Decision Trees**
- **Support Vector Machines (SVM)**

Each classifier is trained on the training data, and the performance is evaluated on the test set.

## Model Evaluation
- **Accuracy**: The accuracy of each model is evaluated on the test set.

Model Comparison Table
Model	            Train Time (s)	Train Accuracy	Test Accuracy
Logistic Regression	0.528586	    0.910319	    0.910415
KNN	                0.010968	    0.926646	    0.896091
Decision Tree	    0.236368	    1.000000	    0.889415
SVM	                20.601908	    0.910076	    0.905681
This final summary highlights the strengths and weaknesses of each model, offering guidance on which to deploy depending on the specific business requirements.


## Findings and Actionable Insights
- Key findings from the analysis are presented, including model performance comparisons.
- Actionable insights for the marketing team are provided based on the predictive models' performance, helping to improve campaign targeting.

## Next Steps and Recommendations
- **Model Improvement**: Notyable Suggestions for hyperparameter tuning to improve accuracy and ROC AUC scores.
- **Feature Engineering**: Explored additional feature engineering to further improve model performance.
- **Business Application**: Recommended for deploying the selected model into a production environment, with regular monitoring and retraining to maintain performance.
- **Further Analysis**: Investigated the impact of additional features or external data sources on model performance to enhance predictions.

