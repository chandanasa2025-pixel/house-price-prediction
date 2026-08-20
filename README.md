# house-price-prediction
ML project for house price prediction
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np

data=fetch_california_housing(as_frame=True)
X=data.data
y=data.target

print("Dataset:")
print(X.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:",mae)
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)

sample=X_test.iloc[0:5]
predictions=model.predict(sample)

print("\nSample Predictions:")
for actual,predicted in zip(y_test.iloc[0:5],predictions):
    print("Actual:",actual,"Predicted:",predicted)
