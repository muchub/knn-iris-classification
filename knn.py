import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset = 'dataset/iris.csv'
iris = pd.DataFrame(pd.read_csv(dataset))
iris_col, iris_class = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
user_input_data = []

k_input = int(input("Enter K Neighbor value: "))
for i in iris_class:
    input_data = float(input("Please enter " + i + ": "))
    user_input_data.append(input_data)

X, y = iris[iris_col].values, iris['species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=k_input)

knn_classifier.fit(X_train, y_train)

y_pred_test = knn_classifier.predict(X_test)
y_pred_train = knn_classifier.predict(X_train)
y_pred_user_input = knn_classifier.predict(np.array([user_input_data]))

test_data_accuracy = accuracy_score(y_test, y_pred_test)
train_data_accuracy = accuracy_score(y_train, y_pred_train)

print(f'\nTest data Accuracy: {test_data_accuracy:.2f}')
print(f'Train data Accuracy: {train_data_accuracy:.2f}')

print('\nClassification Report Test Data:')
print(classification_report(y_test, y_pred_test))

print('\nClassification Report Train Data:')
print(classification_report(y_train, y_pred_train))

print(f"\nBased on your input KNN predict '{y_pred_user_input[0]}'\n")