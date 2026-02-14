#assignment_question_five.py

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

data_frame = pd.read_csv("kidney_disease.csv").dropna() #drops na values

#convert text into numbers
#NOTE: Gemini was used here to debug errors..
label_encoder = LabelEncoder()
for col in data_frame.columns:
    if not pd.api.types.is_numeric_dtype(data_frame[col]):
        data_frame[col] = label_encoder.fit_transform(data_frame[col].astype(str))
##Gemini code ends here

#feature matrix X without classification
X = data_frame.drop('classification', axis=1)

#creating vector Y with only classification column
y = data_frame['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50)

k_values = [1,3,5,7,9]

#finding best k algorithm
#test k values, store k:accuracy values in a dictionary, compare
results = {}
best_k = -1
best_accuracy = -1.0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[k] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

    print(f"k-value: {k} with accuracy {accuracy}")

print(f"Best k value: {best_k} with accuracy {best_accuracy}")


#In this case, changing the k does not really affect the accuracy, even with higher random data.
#However, in general, a KNN model with a much lower k value is more prone to overfitting, whereas a much higher
#k value is more prone to underfitting

#A much lower k value is more prone to overfitting because it basically grabs a small amount of neighbors,
#and the model will basically "memorize" the training data. This means that it would not be able to generalize
#with any new data since it is following the training data too closely.

#Meanwhile, very large k values is more prone to underfitting since it grabs too many neighbors, and it does not find any
#patterns; the model is not learning anything. It simply just overgeneralizes the data and fails to capture
#the specific details of it, therefore it is too simple.
