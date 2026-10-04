import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("cancer.csv")

data = data.drop(['id', 'Unnamed: 32'], axis=1)
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]

x_data = data.drop(['diagnosis'], axis=1)
y_data = data['diagnosis']

# data normalize
x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())

x_train, x_test, y_train, y_test = train_test_split (x_data, y_data, 
                                                     test_size= 0.15, 
                                                     random_state= 42)
model = LogisticRegression()
model.maxiter = 1000000
model.fit(x_train, y_train)

out = model.predict(x_test)

compare_out = y_test == out
compare_out.reset_index()

counter = 0
for i in compare_out:
    if i == False:
        counter += 1

print (counter, "/", len(compare_out))
print (100 - (counter*100 / len(compare_out)))
