# 🩺 Cancer Classification using Logistic Regression

A Machine Learning project that classifies cancer cases as malignant or benign using Logistic Regression with data normalization and model accuracy evaluation.


## Overview

This project uses a cancer dataset to classify cases into two categories: **malignant (M)** and **benign (B)**.

The dataset is prepared by removing unnecessary columns, converting the diagnosis labels into numerical values, and normalizing the input features. Logistic Regression is then trained on the processed data and used to predict the diagnosis of unseen test samples.

The final accuracy is calculated by comparing the predicted results with the actual diagnosis labels.


## Features

* Cancer classification
* Data preprocessing
* Removal of unnecessary columns
* Conversion of diagnosis labels into numerical values
* Feature normalization
* Train/test data splitting
* Logistic Regression classification
* Prediction on test data
* Accuracy evaluation


## Technologies Used

* Python
* Pandas
* Scikit-learn


## Dataset

The project uses the following dataset:

```text id="g8y4sl"
cancer.csv
```

The dataset contains numerical features related to cancer cases.

The target variable is:

```text id="j5m2rx"
diagnosis
```

The diagnosis values are converted into numerical labels:

```text id="n3c7qw"
M → 1
B → 0
```

where `M` represents a malignant case and `B` represents a benign case.


## Data Preprocessing

Several preprocessing steps are performed before training the model.

### Removing Unnecessary Columns

The `id` and `Unnamed: 32` columns are removed because they are not used as input features.

```python id="p2y6vk"
data = data.drop(['id', 'Unnamed: 32'], axis=1)
```

### Encoding Diagnosis Labels

The diagnosis values are converted into binary numerical values:

```python id="c9x4mw"
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]
```

This allows the Logistic Regression model to work with numerical target values.

### Separating Features and Target

The input features and target variable are separated:

```python id="z7t1qp"
x_data = data.drop(['diagnosis'], axis=1)
y_data = data['diagnosis']
```

### Data Normalization

The input features are normalized using Min-Max normalization:

```python id="u5n8ad"
x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())
```

This scales the feature values to a common range.

### Train/Test Split

The dataset is divided into training and testing sets using a 85/15 split.

```text id="m4x7pk"
85% → Training data
15% → Testing data
```


## Machine Learning Workflow

The project follows the workflow below:

```text id="r7v3zn"
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

### Logistic Regression

Logistic Regression is used as the classification algorithm for this project.

```python id="h2k9wf"
model = LogisticRegression()
model.fit(x_train, y_train)
```

After training, the model predicts the diagnosis of the test samples:

```python id="v6p3zs"
out = model.predict(x_test)
```


## Model Training

The Logistic Regression model is trained using the training dataset:

```python id="q8m1xc"
model.fit(x_train, y_train)
```

The trained model is then used to predict the classes of the test dataset.


## Evaluation

The predictions are compared with the actual test labels.

```python id="n5w2lr"
compare_out = y_test == out
```

The project counts the number of incorrect predictions and then calculates the classification accuracy.

```python id="d4k7yp"
print(counter, "/", len(compare_out))
print(100 - (counter*100 / len(compare_out)))
```

The final result represents the percentage of correctly classified test samples.

### Example Output

```text id="p9x3kt"
XX / XXX
XX.XX
```

The exact accuracy depends on the dataset and the resulting train/test split.


## Project Structure

```text id="s6v2qm"
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

```bash id="w3k8mf"
pip install pandas scikit-learn
```

You can also install all dependencies using:

```bash id="c7r5yn"
pip install -r requirements.txt
```



## Future Improvements

* Use a fixed `random_state` for reproducible results
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
