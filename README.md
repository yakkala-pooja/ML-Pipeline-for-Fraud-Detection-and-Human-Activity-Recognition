# ML-Pipeline-for-Fraud-Detection-and-Human-Activity-Recognition

This project implements two robust machine learning pipelines to solve real-world classification problems:

- **Fraud Detection**: Classifying transactions as fraudulent or legitimate.
- **Human Activity Recognition**: Identifying different types of physical activities based on sensor data.

The pipelines involve advanced preprocessing, class imbalance handling, model training with cross-validation, hyperparameter tuning, and performance evaluation. Trained models are saved for later deployment or testing on unseen data.

---

## Key Features

- End-to-end ML pipelines for binary and multi-class classification  
- Handles severe **class imbalance** using **SMOTE-ENN** and **SMOTE-Tomek** techniques  
- Uses **Stratified K-Fold Cross-Validation** and **GridSearchCV** for robust training  
- Includes **XGBoost**, **Random Forest**, and a **Stacked Ensemble with Logistic Regression**  
- Saves models using `joblib` for easy reuse  
- Evaluation metrics and confusion matrices help understand model performance  

---

## Repository Structure

```
.
├── FraudDetection.ipynb            # Fraud detection model pipeline
├── HAR.ipynb                       # Activity recognition with ensemble model
├── Testing.ipynb                   # Loads saved models, runs tests, and evaluates performance
├── README.md                       # This file
├── Report.pdf                      # Report of all the results and observations
├── requirements.txt                # Required packages file
```

---

## Model Architectures

### 1. Fraud Detection (`FD.joblib`)
- **Model**: XGBoost Classifier
- **Pipeline**:
  - Load & preprocess transaction data
  - Encode categorical labels
  - Balance classes using **SMOTE-ENN**
  - Train model with **Stratified K-Fold CV**
  - Tune hyperparameters with **GridSearchCV**
  - Evaluate with accuracy, confusion matrix, and classification report

### 2. Human Activity Recognition (`HAR.joblib`)
- **Model**: Stacked Ensemble (Random Forest + XGBoost → Logistic Regression)
- **Pipeline**:
  - Load sensor data and encode labels
  - Balance dataset using **SMOTE-Tomek**
  - Train **Random Forest** and **XGBoost**
  - Create a stacking classifier with **Logistic Regression** as the meta-learner
  - Evaluate using **macro F1 score** and confusion matrices

---

## Model Testing

Run `Testing.ipynb` to:
- Load both saved models
- Apply the same preprocessing to test datasets
- Generate predictions
- Visualize performance (classification reports, confusion matrices, etc.)

---

## Installation

Requires **Python 3.10+**. Install dependencies via pip:

```bash
pip install scikit-learn xgboost imbalanced-learn pandas numpy joblib
```

---

## Getting Started

1. Clone this repository  
2. (Optional) Open and rerun `FraudDetection.ipynb` or `HAR.ipynb` to retrain models  
3. Run `Testing.ipynb` to test model performance on new data
4. You can find the dataset in 'https://www.kaggle.com/datasets/poojayakkala/fraud-detection-and-human-activity-recognition'

You can also load and use the models in other applications:

```python
import joblib

# Load saved models
fraud_model = joblib.load('FD.joblib')
activity_model = joblib.load('HAR.joblib')
```

---

## Example Use Cases

- **Fraud Detection**: For fintech applications, detecting anomalies in transaction behavior  
- **Activity Recognition**: For healthcare or fitness apps, understanding user behavior from wearable devices  

---

## Requirements

- Python 3.10+
- `scikit-learn`
- `xgboost`
- `imbalanced-learn`
- `pandas`, `numpy`
- `joblib`

---

## Contact

Feel free to reach out for suggestions, improvements, or collaborations!
