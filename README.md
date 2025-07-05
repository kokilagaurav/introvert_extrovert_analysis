# Introvert vs Extrovert Personality Analysis

This project analyzes personality traits to classify individuals as introverts or extroverts using machine learning techniques. The analysis uses survey data with various behavioral and psychological indicators.

## Project Overview

The goal is to build a predictive model that can classify personality types based on behavioral patterns such as time spent alone, social event attendance, social media activity, and other personality indicators.

## Dataset

The dataset contains the following features:
- **Time_spent_Alone**: Hours spent alone per day
- **Social_event_attendance**: Frequency of attending social events
- **Going_outside**: Frequency of going outside
- **Friends_circle_size**: Size of friend circle
- **Post_frequency**: Social media posting frequency
- **Stage_fear**: Whether the person has stage fear (categorical)
- **Drained_after_socializing**: Whether the person feels drained after socializing (categorical)
- **Personality**: Target variable (Introvert/Extrovert)

## Data Preprocessing

### 1. Data Cleaning
- Removed ID column as it's not relevant for prediction
- Handled missing values:
  - Numerical features: Filled with median values
  - Categorical features: Filled with mode values

### 2. Feature Engineering
- **Numerical Features**: Standardized using StandardScaler
- **Categorical Features**: One-hot encoded using OneHotEncoder
- **Target Variable**: Label encoded (Extrovert=0, Introvert=1)

### 3. Class Imbalance Handling
- Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset
- Achieved balanced class distribution for better model performance

## Exploratory Data Analysis

### Key Visualizations Created:
1. **Distribution Analysis**: Histograms and KDE plots for numerical features
2. **Personality Distribution**: Pie chart showing class distribution
3. **Categorical Analysis**: Count plots for stage fear and social draining vs personality
4. **Correlation Analysis**: Heatmap showing feature correlations
5. **Outlier Detection**: Box plots for identifying outliers

### Key Insights:
- Dataset shows class imbalance between introverts and extroverts
- Strong correlations exist between certain behavioral patterns and personality types
- Stage fear and feeling drained after socializing are strong indicators

## Model Development

### Architecture
- **Neural Network**: Deep learning model using TensorFlow/Keras
- **Layers**:
  - Input layer with L2 regularization
  - Hidden layers with batch normalization and dropout
  - Output layer with sigmoid activation for binary classification

### Model Configuration:
```python
- Dense(32, activation='relu', L2 regularization=0.01)
- BatchNormalization()
- Dropout(0.5)
- Dense(16, activation='relu', L2 regularization=0.01)
- Dropout(0.5)
- Dense(1, activation='sigmoid')
```

### Training Strategy:
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Callbacks**: 
  - Early Stopping (patience=5)
  - Learning Rate Reduction (factor=0.5)
- **Validation**: 5-fold Stratified Cross-Validation

## Model Performance

### Cross-Validation Results:
- **Mean CV Accuracy**: 95.69% (±0.40%)
- **Individual Fold Scores**: [95.67%, 95.78%, 95.92%, 95.69%, 95.39%]

### Test Set Performance:
- **Test Accuracy**: High performance maintained on unseen data
- **Generalization**: No overfitting detected (CV-Test difference < 0.05)

### Evaluation Metrics:
- **Precision**: High precision for both classes
- **Recall**: Balanced recall across personality types
- **F1-Score**: Strong F1 scores indicating robust performance
- **Confusion Matrix**: Detailed classification results

## Files Structure

```
d:\projects\Introvert_extrovert_analysis\
├── notebook.ipynb           # Main analysis notebook
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── actual_prediction2.csv  # Model predictions
├── submission.csv          # Final submission file
└── README.md              # This file
```

## Key Libraries Used

- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, tensorflow/keras
- **Imbalanced Learning**: imbalanced-learn (SMOTE)
- **Statistical Analysis**: scipy

## Usage

1. **Data Preparation**:
   ```python
   # Load and preprocess data
   df = pd.read_csv('train.csv')
   # Handle missing values and feature engineering
   ```

2. **Model Training**:
   ```python
   # Apply SMOTE for class balancing
   # Train neural network with cross-validation
   ```

3. **Prediction**:
   ```python
   # Load test data and make predictions
   # Generate submission file
   ```

## Results and Insights

The model successfully achieves high accuracy in classifying personality types, with key behavioral indicators being:
- Time spent alone
- Social event attendance patterns
- Response to social situations (stage fear, feeling drained)
- Social media activity patterns

The balanced approach using SMOTE and robust cross-validation ensures reliable performance on unseen data.

## Future Improvements

- Feature engineering with interaction terms
- Ensemble methods combining multiple algorithms
- Additional behavioral indicators
- Hyperparameter tuning with grid search
- Model interpretability analysis

## Installation

```bash
pip install pandas numpy scikit-learn tensorflow seaborn matplotlib imbalanced-learn scipy
```

## Contributing

Feel free to contribute by:
- Adding new features
- Improving model performance
- Enhancing visualizations
- Adding more evaluation metrics