# Machine Learning Labs

This repository contains my Machine Learning laboratory practicals and assignments.

## Student Details

- **Name:** Shubh Veerwani
- **Roll No.:** 34
- **Section:** B
- **Batch:** B3

## Practicals at a Glance

| Lab | Topic | Dataset | Notebook |
| --- | ----- | ------- | -------- |
| 1 | Data Preprocessing | `Titanic-Dataset.csv` | `ML-PL1.ipynb` |
| 2 | Regression | USA Housing dataset | `ML-PL2.ipynb` |
| 3 | Concept Learning (Find-S, Candidate Elimination) | `enjoy.csv` | `ML-PL3.ipynb` |
| 4 | Decision Tree Classification | `winequality-red.csv` | `ML-PL4.ipynb` |
| 5 | K-Nearest Neighbours (KNN) | `diabetes.csv` | `Lab5_KNN_Diabetes_Complete.ipynb` |
| 6 | Artificial Neural Network (ANN) | `seeds_new.csv` | `ML-PL6.ipynb` |
| 7 | Naïve Bayes Classifier | `Raisin_Dataset.xlsx` | `ML-Lab-7.ipynb` |
| 8 | Support Vector Machine (SVM) | `ai4i2020.csv` | `ML-PL8.ipynb` |

## Lab Practicals

### Lab 1 - Data Preprocessing

**Topic:** Data Preprocessing on Titanic Dataset

- Dataset: Titanic-Dataset.csv
- Missing value handling
- Outlier detection and removal using IQR
- Data encoding
- Univariate, Bivariate and Multivariate Analysis
- Feature Scaling
- Train-Test Split

**Files:**

- `ML-PL1.ipynb`
- `ML-PL1.py`
- `Titanic-Dataset.csv`

---

### Lab 2 - Regression

**Topic:** USA Housing Dataset - Regression

- Data preprocessing and analysis
- Simple Linear Regression
- Multiple Linear Regression
- Ridge Regression
- Lasso Regression
- Model evaluation and comparison

**Files:**

- `ML-PL2.ipynb`
- `ML-PL2.py`
- USA Housing dataset

---

### Lab 3 - Concept Learning

**Topic:** S-Algorithm and Candidate Elimination Algorithm

- Concept learning
- S-Algorithm / Find-S
- Candidate Elimination Algorithm
- Specific Boundary
- General Boundary
- Version Space

**Dataset:**

- `enjoy.csv`

**Files:**

- `ML-PL3.ipynb`
- `ML-PL3.py`
- `enjoy.csv`

---

### Lab 4 - Classification

**Topic:** Wine Quality Analysis and Decision Tree Classification

- Wine Quality dataset analysis
- Univariate, Bivariate and Multivariate Analysis
- Outlier analysis using IQR
- Decision Tree Classifier
- Entropy criterion
- Confusion Matrix
- Accuracy, Precision, Recall and F1-Score
- ROC Curve and ROC-AUC
- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Learning Curve
- Wine quality prediction

**Dataset:**

- `winequality-red.csv`

**Files:**

- `ML-PL4.ipynb`
- `ML-PL4.py`
- `winequality-red.csv`

---

### Lab 5 - K-Nearest Neighbours (KNN)

**Topic:** Diabetes Prediction using KNN

- Dataset inspection (768 records, 8 features, binary Outcome)
- Detection of hidden missing values (invalid zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI)
- Univariate, Bivariate and Multivariate Analysis
- Outlier detection using IQR
- Data cleaning using median imputation
- Stratified train-test split and feature scaling using StandardScaler
- KNN classifier (K = 5)
- Accuracy, Precision, Recall, F1-Score and Confusion Matrix
- Best K selection using the train/test accuracy curve and 10-fold stratified cross-validation
- Hyperparameter tuning using GridSearchCV (n_neighbors, weights, metric)
- ROC Curve and ROC-AUC
- Diabetes prediction for a new patient

**Key results:**

- KNN (K = 5): accuracy 0.7532, F1-score 0.6346, ROC-AUC 0.7886
- Tuned KNN (K = 19, Euclidean, uniform weights): accuracy 0.7468, F1-score 0.6061, ROC-AUC 0.8077

**Dataset:**

- `diabetes.csv`

**Files:**

- `Lab5_KNN_Diabetes_Complete.ipynb`
- `diabetes.csv`

---

### Lab 6 - Artificial Neural Network (ANN)

**Topic:** Seed Classification using a Feed-Forward ANN

- Dataset cleaning (removal of empty and unnamed columns)
- Univariate, Bivariate and Multivariate Analysis
- Outlier detection using IQR
- Feature scaling using StandardScaler
- Feed-forward ANN built with TensorFlow / Keras (ReLU hidden layers, softmax output)
- Training with EarlyStopping
- Accuracy, Precision, Recall, F1-Score and Confusion Matrix

**Dataset:**

- `seeds_new.csv`

**Files:**

- `ML-PL6.ipynb`
- `seeds_new.csv`

---

### Lab 7 - Naïve Bayes Classifier

**Topic:** Raisin Classification (Kecimen and Besni) using Gaussian Naïve Bayes

- Dataset inspection (900 records, numeric shape and size measurements, class label)
- Univariate, Bivariate and Multivariate Analysis
- Outlier detection using IQR
- Class label encoding using LabelEncoder
- Stratified 80/20 train-test split (720 training and 180 testing samples)
- Gaussian Naïve Bayes classifier
- Accuracy, Precision, Recall, F1-Score and Confusion Matrix
- Training versus testing accuracy comparison

**Key results:**

- Test accuracy 0.8278, precision 0.7658, recall 0.9444, F1-score 0.8458
- Confusion matrix [[64, 26], [5, 85]]; training accuracy 0.8375

**Dataset:**

- `Raisin_Dataset.xlsx`

**Files:**

- `ML-Lab-7.ipynb`
- `Raisin_Dataset.xlsx`

---

### Lab 8 - Support Vector Machine (SVM)

**Topic:** Machine Failure Prediction using SVM (AI4I 2020 Predictive Maintenance Dataset)

- Dataset inspection (10,000 records, 14 columns) and missing value / duplicate checks
- Class imbalance analysis (3.39% machine failures)
- Univariate, Bivariate and Multivariate Analysis
- Outlier detection using IQR and failure-mode analysis
- Preprocessing with ColumnTransformer and Pipeline (StandardScaler and OneHotEncoder)
- SVM with Linear kernel and RBF kernel
- Accuracy, Precision, Recall, F1-Score and Confusion Matrix
- Kernel comparison
- Hyperparameter tuning using GridSearchCV (C and kernel, 3-fold, F1 scoring)

**Key results:**

- Linear SVM: accuracy 0.966, F1-score 0.000 (predicts No Failure for every sample)
- RBF SVM: accuracy 0.972, F1-score 0.333
- Tuned SVM (C = 10, RBF): accuracy 0.979, precision 0.771, recall 0.544, F1-score 0.638

**Dataset:**

- `ai4i2020.csv`

**Files:**

- `ML-PL8.ipynb`
- `ai4i2020.csv`

---

## Tools and Technologies

- Python
- Jupyter Notebook
- VS Code
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras (Lab 6)
- openpyxl (Lab 7, to read the Excel dataset)

## Repository Structure

```
machine-learning-labs/
│
├── README.md
│
├── ML-PL1.ipynb
├── ML-PL1.py
├── Titanic-Dataset.csv
│
├── ML-PL2.ipynb
├── ML-PL2.py
│
├── ML-PL3.ipynb
├── ML-PL3.py
├── enjoy.csv
│
├── ML-PL4.ipynb
├── ML-PL4.py
├── winequality-red.csv
│
├── Lab5_KNN_Diabetes_Complete.ipynb
├── diabetes.csv
│
├── ML-PL6.ipynb
├── seeds_new.csv
│
├── ML-Lab-7.ipynb
├── Raisin_Dataset.xlsx
│
├── ML-PL8.ipynb
└── ai4i2020.csv
```

## How to Run

1. Clone the repository: `git clone https://github.com/Shubhveer/machine-learning-labs.git`
2. Install the libraries: `pip install pandas numpy matplotlib seaborn scikit-learn tensorflow openpyxl jupyter`
3. Keep the notebook and its dataset file in the same folder.
4. Open the notebook in Jupyter Notebook, VS Code or Google Colab and run all cells in order.
