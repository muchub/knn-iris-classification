import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset = 'dataset/iris.csv'
iris = pd.DataFrame(pd.read_csv(dataset))
iris_col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_class = []

k_input = int(input("Enter K value: "))

for i_class in iris['species']:
    if i_class in iris_class:
        pass
    else:
        iris_class.append(i_class)

X, y = iris[iris_col].values, iris['species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=k_input)

knn_classifier.fit(X_train, y_train)

y_pred_test = knn_classifier.predict(X_test)
y_pred_train = knn_classifier.predict(X_train)

test_data_accuracy = accuracy_score(y_test, y_pred_test)
train_data_accuracy = accuracy_score(y_train, y_pred_train)

print(f'Test Accuracy: {test_data_accuracy:.2f}')
print(f'Train Accuracy: {train_data_accuracy:.2f}')

print('\nClassification Report Test Data:')
print(classification_report(y_test, y_pred_test))

print('\nClassification Report Train Data:')
print(classification_report(y_train, y_pred_train))