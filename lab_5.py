import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# 
data = np.loadtxt("kinematic_features.txt")
X = data 
# y = np.concatenate((np.zeros(41), np.ones(55)))  # Assuming 41 samples for each class
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)
# # Initialize and train the KNN classifier
# knn = KNeighborsClassifier(n_neighbors=5)  # You can choose an appropriate value for k
# knn.fit(X_train, y_train)
# 
# # Predict on training and test data
# y_train_pred = knn.predict(X_train)
# y_test_pred = knn.predict(X_test)
# 
# # Evaluate confusion matrix and performance metrics
# conf_matrix_train = confusion_matrix(y_train, y_train_pred)
# conf_matrix_test = confusion_matrix(y_test, y_test_pred)
# 
# # Print confusion matrix and classification report for training data
# print("Confusion Matrix (Training Data):")
# print(conf_matrix_train)
# print("\nClassification Report (Training Data):")
# print(classification_report(y_train, y_train_pred))
# 
# # Print confusion matrix and classification report for test data
# print("Confusion Matrix (Test Data):")
# print(conf_matrix_test)
# print("\nClassification Report (Test Data):")
# print(classification_report(y_test, y_test_pred))
# 
# # for the ouput of 90% accuracy, it is incudred that the data is regular fit
# 
# X=np.random.randint(10,size=20)
# Y=np.random.randint(10,size=20)
# data=[[i,j] for i,j in zip(X,Y)]
# classes = ['red' if i[0] + i[1] < 10 else 'blue' for i in data]
# print(data,classes)
# plt.scatter(X,Y,c=classes)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Scatter Plot of Data with Classes')
# plt.show()
# 
# X_test = np.arange(0, 10.1, 0.1)  
# Y_test = np.arange(0, 10.1, 0.1)
# test_data = [[i, j] for i in X_test for j in Y_test]
# 
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(data,classes)
# y_pred = knn.predict(test_data)
# colors = np.where(y_pred == 'red', 'red', 'blue')  # Assigning colors based on predicted classes
# plt.scatter([point[0] for point in test_data], [point[1] for point in test_data], c=colors, alpha=0.1)
# plt.show()
# 
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(data,classes)
# y_pred = knn.predict(test_data)
# colors = np.where(y_pred == 'red', 'red', 'blue')  # Assigning colors based on predicted classes
# plt.scatter([point[0] for point in test_data], [point[1] for point in test_data], c=colors, alpha=0.1)
# plt.show()
# 
# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(data,classes)
# y_pred = knn.predict(test_data)
# colors = np.where(y_pred == 'red', 'red', 'blue')  # Assigning colors based on predicted classes
# plt.scatter([point[0] for point in test_data], [point[1] for point in test_data], c=colors, alpha=0.1)
# plt.show()
# 
# knn = KNeighborsClassifier(n_neighbors=2)
# knn.fit(data,classes)
# y_pred = knn.predict(test_data)
# colors = np.where(y_pred == 'red', 'red', 'blue')  # Assigning colors based on predicted classes
# plt.scatter([point[0] for point in test_data], [point[1] for point in test_data], c=colors, alpha=0.1)
# plt.show()

X = data[:,:2]
print(X.shape)
y = np.concatenate((np.zeros(41), np.ones(55)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
colors = np.where(y_pred == 0, 'red', 'blue')  # Assigning colors based on predicted classes
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, alpha=0.1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Test Data with Predicted Classes')
plt.show()


# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)
# colors = np.where(y_pred == 0, 'red', 'blue')  # Assigning colors based on predicted classes
# plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, alpha=0.1)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Scatter Plot of Test Data with Predicted Classes')
# plt.show()
# 
# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)
# colors = np.where(y_pred == 0, 'red', 'blue')  # Assigning colors based on predicted classes
# plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, alpha=0.1)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Scatter Plot of Test Data with Predicted Classes')
# plt.show()
# 
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)
# colors = np.where(y_pred == 0, 'red', 'blue')  # Assigning colors based on predicted classes
# plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, alpha=0.1)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Scatter Plot of Test Data with Predicted Classes')
# plt.show()
# 

param_grid = {
    'n_neighbors': range(1, 21),  
}

search= KNeighborsClassifier()
grid_search = GridSearchCV(search, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

random_search = RandomizedSearchCV(search, param_distributions=param_grid, n_iter=100, cv=5, random_state=42)
random_search.fit(X_train, y_train)
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print("Best parameters using GridSearchCV:", best_params)
print("Best score using GridSearchCV:", best_score)
print("Best parameters using RandomizedSearchCV:", best_params_random)
print("Best score using RandomizedSearchCV:", best_score_random)