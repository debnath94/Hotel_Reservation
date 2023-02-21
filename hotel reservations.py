# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 10:59:36 2023

@author: debna
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib. pylab as plt
from sklearn. model_selection import train_test_split

df = pd. read_csv("E:/LiveProject/Hotel Reservations.csv/Hotel Reservations.csv")

df. info()
df. shape

duplicate = df. duplicated()
sum(duplicate)

sns. boxplot(df. avg_price_per_room)

from feature_engine. outliers import Winsorizer
winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = ["avg_price_per_room"])

df["avg_price_per_room"] = winsor. fit_transform(df[["avg_price_per_room"]])

#checking variance
df. var()

df. var() == 0

df = df. drop(["required_car_parking_space","repeated_guest","Booking_ID"], axis = 1)

df. isna(). sum()

df. booking_status. value_counts(normalize = True)

df = pd. get_dummies(df, columns = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"], drop_first=True)

#input output split
predictors = df. loc[:, df. columns!="booking_status"]
type(predictors)
target = df["booking_status"]
type(target)

desc = df. describe()

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state = 0)

from sklearn. tree import DecisionTreeClassifier as DT
model = DT(criterion="entropy")
model. fit(x_train, y_train)

#prediction on test data
test_pred = model. predict(x_test)

from sklearn. metrics import accuracy_score
print(accuracy_score(y_test, test_pred))
#np. mean(preds==y_test)

pd. crosstab(y_test, test_pred, rownames = ["Actual"], colnames = ["predictions"])

# Prediction on Train Data
preds = model.predict(x_train)
pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])

print(accuracy_score(y_train, preds))

#np. mean(preds==y_train)

# let us try random forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, rf_clf.predict(x_test))
accuracy_score(y_test, rf_clf.predict(x_test))

confusion_matrix(y_train, rf_clf.predict(x_train))

accuracy_score(y_train, rf_clf.predict(x_train))

#Hyperparameter Tuning
# Creating new model testing with new parameters
forest_new = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')  # n_estimators is the number of decision trees
forest_new.fit(x_train, y_train)

print('Train accuracy: {}'.format(forest_new.score(x_train, y_train)))
print('Test accuracy: {}'.format(forest_new.score(x_test, y_test)))

##train and test score are 0.87 and 0.87








