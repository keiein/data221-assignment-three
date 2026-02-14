#assignment_question_three.py

import pandas as pd
from sklearn.model_selection import train_test_split

data_frame = pd.read_csv("kidney_disease.csv")

#feature matrix X without classification
X = data_frame.drop('classification', axis=1)

#creating vector Y with only classification column
y = data_frame['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)

#We should not train and test a model on the same data as it may
#follow the training data too closely, and almost "memorize" the data.
#It would be a decent way to model training data, but it does not know how to
#react to new data.
#It is called overfitting, and it does not generalize new data.

#A testing set is basically simulating new data, so we can see how the model can perform
#without any bias, and we can identify any overfitting.