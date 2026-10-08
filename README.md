# 🩺 Breast Cancer Classification using Machine Learning

A Machine Learning project containing two implementations of breast cancer classification using the same dataset: a Logistic Regression model in Python and a Jupyter Notebook comparing six different classification algorithms.


## Overview

This repository contains two related implementations for classifying breast cancer cases as **malignant** or **benign**.

Both implementations use the **same Breast Cancer Wisconsin Diagnostic dataset**, but the dataset is loaded in two different ways.

The first implementation is a Python script that uses a downloaded `cancer.csv` file. It performs data preprocessing, feature normalization, and uses **Logistic Regression** for classification.

The second implementation is a Jupyter Notebook that loads the same dataset directly from Scikit-learn using `load_breast_cancer()`. It then compares six different Machine Learning classification algorithms.

The purpose of having both implementations in the same repository is to demonstrate different ways of working with the same dataset and comparing different Machine Learning approaches.


## Projects

### 1. Logistic Regression

The first implementation is written in Python:

```text
cancer_classification.py
```

This project uses the downloaded dataset:

```text
cancer.csv
```

The main steps include:

* Loading the dataset from a CSV file
* Removing unnecessary columns
* Converting diagnosis labels into numerical values
* Separating features and target
* Normalizing the input features
* Splitting the dataset into training and testing sets
* Training a Logistic Regression model
* Predicting the diagnosis of test samples
* Calculating classification accuracy

### 2. Machine Learning Model Comparison

The second implementation is provided as a Jupyter Notebook:

```text
breast_cancer_classification.ipynb
```

Instead of reading a local CSV file, the notebook loads the dataset directly from Scikit-learn:

```python
from sklearn import datasets

cancer = datasets.load_breast_cancer()
```

The notebook then trains and compares six different classification algorithms:

* Support Vector Machine (SVM)
* Gaussian Naive Bayes
* Decision Tree
* Random Forest
* XGBoost
* K-Nearest Neighbors (KNN)

Each model is evaluated using the same training and testing data.


## Features

* Breast cancer classification
* Malignant and benign classification
* Data preprocessing
* Feature normalization
* Train/test data splitting
* Logistic Regression
* Support Vector Machine
* Gaussian Naive Bayes
* Decision Tree
* Random Forest
* XGBoost
* K-Nearest Neighbors
* Accuracy evaluation
* Machine Learning model comparison


## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* Jupyter Notebook


## Dataset

Both projects use the **same Breast Cancer Wisconsin Diagnostic dataset**.

The dataset contains numerical features describing characteristics of breast cancer cell nuclei and is used to classify cases into two categories:

```text
Malignant
Benign
```

### Python Dataset

The Python implementation uses a downloaded copy of the dataset:

```text
cancer.csv
```

The `diagnosis` column is converted into numerical labels:

```text
M → 1
B → 0
```

The columns `id` and `Unnamed: 32` are removed before training.

### Jupyter Notebook Dataset

The notebook loads the dataset directly from Scikit-learn:

```python
cancer = datasets.load_breast_cancer()
```

This avoids manually storing the dataset as a CSV file and provides the dataset directly through the Scikit-learn library.

Therefore, both implementations work with the **same dataset**, but the source of the data in the code is different.


## Data Preprocessing

### Python Implementation

The first implementation performs the following preprocessing steps:

```text
Load cancer.csv
      ↓
Remove unnecessary columns
      ↓
Convert diagnosis labels
      ↓
Separate features and target
      ↓
Normalize features
      ↓
Train/Test Split
```

The features are normalized using Min-Max normalization:

```python
x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())
```

### Jupyter Notebook

The notebook loads the dataset directly from Scikit-learn and separates the features and target:

```python
cancer.data
cancer.target
```

The dataset is then divided into training and testing sets using a 70/30 split.


## Machine Learning Workflow

### Python Project

```text
Load Dataset
      ↓
Data Preprocessing
      ↓
Feature Normalization
      ↓
Train/Test Split
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Accuracy Evaluation
```

### Jupyter Notebook

```text
Downloaded Dataset
      ↓
Data Preprocessing
      ↓
Scikit-learn Dataset
      ↓
Train/Test Split
      ↓
Train Multiple Classifiers
      ↓
Make Predictions
      ↓
Calculate Accuracy
      ↓
Compare Models
```


## Model

Six different Machine Learning classification algorithms are used in this project.

### 1. Support Vector Machine

A Support Vector Machine classifier with a linear kernel is used:

```python
model_SVM = svm.SVC(kernel='linear')
model_SVM.fit(X_train, y_train)
```

The trained model is then used to predict the test data.

### 2. Gaussian Naive Bayes

Gaussian Naive Bayes is used as another classification approach:

```python
model_GNB = GaussianNB()
model_GNB.fit(X_train, y_train)
```

The model predicts the classes of the test samples.

### 3. Decision Tree

A Decision Tree classifier is trained using the training dataset:

```python
model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, y_train)
```


## Model Training

The Logistic Regression model is trained using the training dataset:

```python 
model.fit(x_train, y_train)
```

The trained model is then used to predict the classes of the test dataset.


## Evaluation

The predictions are compared with the actual test labels.

```python 
compare_out = y_test == out
```

The project counts the number of incorrect predictions and then calculates the classification accuracy.

```python 
print(counter, "/", len(compare_out))
print(100 - (counter*100 / len(compare_out)))
```

The final result represents the percentage of correctly classified test samples.

### Example Output

```text
XX / XXX
XX.XX
```

The exact accuracy depends on the dataset and the resulting train/test split.


## Project Structure

```text 
Cancer-Classification/
│
├── cancer.csv
├── cancer_classification.py
├── requirements.txt
├── LICENSE
└── README.md
```


## Installation

Install the required libraries using:

```bash 
pip install pandas scikit-learn
```

You can also install all dependencies using:

```bash 
pip install -r requirements.txt
```


## How to Run

1. Make sure Python is installed.
2. Place `cancer.csv` in the project directory.
3. Install the required libraries.
4. Run the Python script:

```bash 
python cancer_classification.py
```

The program will train the Logistic Regression model, make predictions on the test dataset, and print the number of incorrect predictions and the final accuracy.


## Future Improvements

* Add training and test accuracy separately
* Add a confusion matrix
* Add precision, recall, and F1-score
* Visualize the dataset and feature relationships
* Compare Logistic Regression with other classification algorithms
* Add cross-validation
* Improve the model evaluation process


## Contributing

Contributions are welcome.

You can improve the preprocessing, add new Machine Learning algorithms, improve the evaluation process, or add new visualizations to the project.


## License

This project is licensed under the **MIT License**.


## Author

**Mohammad Reza Bakhshandeh**

Electrical Engineering (Electronics) Graduate

Interested in Industrial Automation, Embedded Systems, PLC Programming, Python Development, Computer Vision, Machine Learning, and Artificial Intelligence.
