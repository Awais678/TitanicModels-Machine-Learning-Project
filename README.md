# Titanic Survival Prediction using Machine Learning

## Overview
This project predicts passenger survival on the Titanic using different Machine Learning classification algorithms. The project includes data preprocessing, feature engineering, model training, and performance evaluation using the Titanic dataset.

The goal is to compare the performance of multiple ML models and identify which model performs best for survival prediction.

---

## Dataset
Dataset used: Titanic Dataset from Seaborn/Kaggle

The dataset contains information about Titanic passengers including:
- Passenger class
- Gender
- Age
- Fare
- Family information
- Embarkation port

Target Variable:
- `survived`
  - 0 = Did Not Survive
  - 1 = Survived

---

## Libraries Used

```python
numpy
pandas
seaborn
matplotlib
scikit-learn
```

---

## Project Workflow

### 1. Data Loading
The Titanic dataset was loaded using Seaborn:

```python
df = sns.load_dataset("titanic")
```

---

### 2. Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns:
  - `deck`
  - `embark_town`
  - `alive`
  - `class`
  - `who`
  - `adult_male`

- Filled missing values in `age` using mean age
- Removed rows with missing `embarked` values
- Encoded categorical variables using `LabelEncoder`
- Converted boolean and categorical columns into integer format

---

### 3. Feature Selection

Input Features (`X`):
- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked
- alone

Target (`y`):
- survived

---

### 4. Train-Test Split

Dataset was divided into:
- 80% Training Data
- 20% Testing Data

```python
train_test_split(test_size=0.2, random_state=42)
```

---

### 5. Feature Scaling

Standardization was applied using `StandardScaler` for models sensitive to feature scaling such as:
- KNN
- SVM
- Decision Tree

---

## Machine Learning Models Used

### 1. Logistic Regression
Accuracy: **80.33%**

#### Classification Report
| Metric | Class 0 | Class 1 |
|---|---|---|
| Precision | 0.85 | 0.74 |
| Recall | 0.83 | 0.77 |
| F1-Score | 0.84 | 0.75 |

---

### 2. K-Nearest Neighbors (KNN)
Accuracy: **78.09%**

#### Classification Report
| Metric | Class 0 | Class 1 |
|---|---|---|
| Precision | 0.82 | 0.72 |
| Recall | 0.83 | 0.71 |
| F1-Score | 0.82 | 0.72 |

---

### 3. Gaussian Naive Bayes
Accuracy: **77.52%**

#### Classification Report
| Metric | Class 0 | Class 1 |
|---|---|---|
| Precision | 0.85 | 0.68 |
| Recall | 0.77 | 0.78 |
| F1-Score | 0.81 | 0.73 |

---

### 4. Decision Tree Classifier
Accuracy: **76.96%**

#### Classification Report
| Metric | Class 0 | Class 1 |
|---|---|---|
| Precision | 0.81 | 0.70 |
| Recall | 0.81 | 0.71 |
| F1-Score | 0.81 | 0.71 |

---

### 5. Support Vector Machine (SVM)
Accuracy: **80.33%**

#### Classification Report
| Metric | Class 0 | Class 1 |
|---|---|---|
| Precision | 0.84 | 0.74 |
| Recall | 0.83 | 0.75 |
| F1-Score | 0.84 | 0.75 |

---

## Model Comparison

| Model | Accuracy |
|---|---|
| Logistic Regression | 80.33% |
| SVM | 80.33% |
| KNN | 78.09% |
| Gaussian Naive Bayes | 77.52% |
| Decision Tree | 76.96% |

---

## Best Performing Models
The best-performing models in this project were:
- Logistic Regression
- Support Vector Machine (SVM)

Both achieved an accuracy of approximately **80.33%**.

---

## Project Structure

```bash
TitanicModels/
│
├── TitanicModels.ipynb
├── README.md
├── requirements.txt
└── dataset/
```

---

## How to Run the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/TitanicModels.git
cd TitanicModels
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

---

## Future Improvements
- Hyperparameter tuning
- Cross-validation
- Random Forest and XGBoost implementation
- Feature engineering improvements
- Model deployment using Flask or FastAPI

---

## Learning Outcomes
This project helped in understanding:
- Data preprocessing techniques
- Handling missing values
- Label encoding
- Feature scaling
- Classification algorithms
- Model evaluation metrics
