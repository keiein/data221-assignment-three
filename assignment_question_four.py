#assignment_question_four.py

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)

knn = KNeighborsClassifier(n_neighbors = 5) #k=5
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test) #make predictions

confusion_matrix_data = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(confusion_matrix_data)
print(accuracy)
print(precision)
print(recall)
print(f1)

#True Positive - Model correctly identified kidney disease
#True Negative - Model correctly identified healthy person
#False Positive - Model predicted that someone healthy has a disease
#False Negative - Model predicted that someone who has the disease was healthy

#Accuracy alone is not enough to evaluate a model because even a high accuracy (like 99%) is
#useless in this scenario since it failed to find the people who are sick.

#Recall is most important if missing a kidney disease is serious because a low recall in this case
#means having too many false negatives (predicting someone who has the disease is healthy) and that means
#we want to increase the recall.
