# 🩺 Breast Cancer Classification using Machine Learning

A Machine Learning project that classifies breast cancer cases as malignant or benign by comparing six different classification algorithms using the Scikit-learn Breast Cancer dataset.


## Overview

This project demonstrates how different Machine Learning classification algorithms can be used to classify breast cancer cases.

The project uses the built-in Breast Cancer dataset provided by Scikit-learn. The dataset contains numerical features describing characteristics of breast cancer cells, with the goal of predicting the cancer diagnosis.

Six different classification algorithms are trained and evaluated using the same training and testing datasets:

* Support Vector Machine (SVM)
* Gaussian Naive Bayes
* Decision Tree
* Random Forest
* XGBoost
* K-Nearest Neighbors (KNN)

The performance of each model is evaluated using **accuracy score**, allowing the results of the different algorithms to be compared.


## Features

* Breast cancer classification
* Built-in Scikit-learn dataset
* Train/test data splitting
* Support Vector Machine classification
* Gaussian Naive Bayes classification
* Decision Tree classification
* Random Forest classification
* XGBoost classification
* K-Nearest Neighbors classification
* Accuracy evaluation
* Comparison of multiple Machine Learning models


## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost


## Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic dataset** provided by Scikit-learn.

The dataset is loaded directly using:

```python
from sklearn import datasets

cancer = datasets.load_breast_cancer()
```

The dataset provides numerical features related to breast cancer cell characteristics.

The target variable contains two classes:

```text
malignant
benign
```

The available feature names and target names are displayed in the notebook.

The dataset contains numerical features related to cancer cases.

The diagnosis values are converted into numerical labels:

```text 
M → 1
B → 0
```

where `M` represents a malignant case and `B` represents a benign case.


## Data Preprocessing

The dataset is loaded directly from Scikit-learn and separated into input features and target labels.

The input data is stored in:

```python
cancer.data
```

and the target labels are stored in:

```python
cancer.target
```

### Removing Unnecessary Columns

The `id` and `Unnamed: 32` columns are removed because they are not used as input features.

```python 
data = data.drop(['id', 'Unnamed: 32'], axis=1)
```

### Encoding Diagnosis Labels

The diagnosis values are converted into binary numerical values:

```python 
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]
```

This allows the Logistic Regression model to work with numerical target values.

### Separating Features and Target

The input features and target variable are separated:

```python 
x_data = data.drop(['diagnosis'], axis=1)
y_data = data['diagnosis']
```

### Data Normalization

The input features are normalized using Min-Max normalization:

```python 
x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())
```

This scales the feature values to a common range.

### Train/Test Split

The dataset is divided into training and testing sets using a 85/15 split.

```text 
85% → Training data
15% → Testing data
```


## Machine Learning Workflow

The project follows the workflow below:

```text 
Cancer Dataset
      ↓
Remove Unnecessary Columns
      ↓
Convert Diagnosis Labels
      ↓
Separate Features and Target
      ↓
Normalize Input Features
      ↓
Train/Test Split
      ↓
Train Logistic Regression Model
      ↓
Make Predictions
      ↓
Compare Predictions with Actual Values
      ↓
Calculate Accuracy
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
