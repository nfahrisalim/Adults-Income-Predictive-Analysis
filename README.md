# Adults Income Predictive Analytics

**Author:** Naufal Fahri

## Overview
This project analyzes the Adults Income dataset to identify and predict factors that contribute to high annual income (>$50K vs ≤$50K). Using machine learning techniques, we aim to understand employment dynamics and income determinants to help governments and individuals make informed decisions.

## Dataset
- **Source:** [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Size:** 48,842 samples with 15 features
- **Target:** Binary classification (income ≤50K or >50K)
- **Missing Values:** ~7% of dataset marked with '?'

## Business Understanding

### Stakeholders
1. **Government** - Policy making for human resource advancement
2. **Individuals** - Career guidance and income improvement insights

### Problem Statements
1. Which features have the most influence on income?
2. Can income be predicted based on personal characteristics?

### Goals
- Identify features with high correlation to income
- Achieve >90% prediction accuracy

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Distribution of categorical and numerical features
- **Multivariate Analysis:** Feature relationships and correlations
- **Key Findings:** Education class, marital status, and relationship are most influential factors

### 2. Data Preparation
- **Feature Encoding:** One-Hot and Ordinal encoding comparison
- **Missing Value Handling:** Two approaches tested:
  - Drop missing values
  - KNN Imputation (k=6)
- **Train-Test Split:** 90/10 ratio
- **Standardization:** Applied to numerical features

### 3. Model Development
**Algorithms Tested:**
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost
- AdaBoost

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Key Features
- **Age, Education Level, Work Hours:** Numerical features
- **Work Class, Education, Marital Status, Occupation:** Categorical features
- **Relationship, Race, Gender, Native Country:** Demographic features
- **Capital Gain/Loss:** Financial features

## Project Structure
```
Adults Income Analysis/
├── Dataset/
│   └── adult.csv
├── Screenshots/
│   ├── confusionMatrix.png
│   ├── corrMat.png
│   └── [various visualization files]
├── PredictiveAnalysis.ipynb
├── convert_csv_autoint.py
└── README.md
```

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
dython
```

## Key Insights
- **Education:** Clear income gap starting at bachelor's degree level
- **Marital Status:** Stable marriages correlate with higher incomes
- **Occupation:** Specialists, managers, and protective services have highest incomes
- **Gender:** Male incomes are higher than female incomes
- **Race:** White and Asian-Pacific Islander groups show highest incomes

## Results
The analysis demonstrates that demographic and educational factors are strong predictors of income levels, providing valuable insights for both policy makers and individuals seeking career advancement.

## Usage
1. Install required dependencies
2. Run `PredictiveAnalysis.ipynb` in Jupyter environment
3. Execute cells sequentially for complete analysis

## Future Work
- Feature engineering optimization
- Advanced ensemble methods
- Deep learning approaches
- Real-time prediction deployment
