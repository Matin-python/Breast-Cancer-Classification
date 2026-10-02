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
