# introvert_extrovert_analysis# Introvert vs Extrovert Personality Prediction

This repository contains my solution for a Kaggle Playground challenge to predict whether a person is an introvert or extrovert based on their personality traits. This is my first Kaggle Playground competition, and I am approaching the problem using deep learning techniques.

## Project Overview

The goal is to classify individuals as introvert or extrovert using features such as:
- Time spent alone
- Social event attendance
- Going outside frequency
- Friends circle size
- Post frequency
- Stage fear
- Drained after socializing

## Current Progress

- **Data Loading & Exploration:**  
  Loaded the training data and performed exploratory data analysis (EDA) using pandas, seaborn, and matplotlib.
- **Missing Value Handling:**  
  Filled missing values in both numerical and categorical columns using median and mode, respectively.
- **Feature Visualization:**  
  Visualized feature distributions, correlations, and relationships with the target variable.
- **Preprocessing:**  
  - Encoded categorical variables using OneHotEncoder.
  - Scaled numerical features using StandardScaler.
  - Used `ColumnTransformer` to combine preprocessing steps.
- **Data Splitting:**  
  Split the data into training and test sets.
- **Test Data Preparation:**  
  Applied the same preprocessing to the test dataset.

## Next Steps

- **Model Building:**  
  Build and train a deep learning model (e.g., using TensorFlow/Keras or PyTorch) for classification.
- **Model Evaluation:**  
  Evaluate the model using appropriate metrics (accuracy, confusion matrix, etc.).
- **Submission:**  
  Generate predictions on the test set and prepare a submission file for Kaggle.
- **Further Improvements:**  
  - Experiment with different model architectures and hyperparameters.
  - Try feature engineering and advanced preprocessing techniques.
  - Analyze feature importance and model interpretability.

## How to Run

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Open `notebook.ipynb` and run the cells step by step.

## Files

- `notebook.ipynb` — Main Jupyter notebook with code and analysis.
- `train.csv` — Training data.
- `test.csv` — Test data for submission.
- `sample_submission.csv` — Sample submission format.

---