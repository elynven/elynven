
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('Advertising (1).csv')

from sklearn.model_selection import train_test_split

X=df.drop('Sales', axis=1)
y=df.Sales.copy()

#training and testing split using all feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2) 

from sklearn.linear_model import LinearRegression
modellr = LinearRegression()
modellr.fit(X_train, y_train)
y_pred = modellr.predict(X_test)

y_pred

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# The mean absolute error
print("Mean absolute error: {} ".format(mean_absolute_error(y_test, y_pred)))

# The mean squared error
print("Mean squared error: {} ".format(mean_squared_error(y_test, y_pred)))

# Root mean squared error
print("Root mean squared error: {} ".format(mean_squared_error(y_test, y_pred)**0.5))

# Explained variance score: 1 is perfect prediction
print('Variance score: {} '.format(r2_score(y_test,y_pred)))

df_prediction = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_prediction

import pickle

pickle.dump(modellr, open("Advertising.h5", "wb")) 

loaded_model = pickle.load(open("Advertising.h5", "rb")) #rb: read binary
new_pred = loaded_model.predict(X_test) # testing (examination)
dfnew_pred = pd.DataFrame({'Actual': y_test, 'Predicted': new_pred})
dfnew_pred
