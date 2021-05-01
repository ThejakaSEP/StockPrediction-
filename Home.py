#importing necessary librabries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Setting a style to the plot ( styles are available in the documentation)
plt.style.use('dark_background')

#Loading the dataset downloaded from Kaggle to a data frame
df = pd.read_csv('/Users/thejakamahaulpatha/PycharmProjects/NetflixStockPrediction/NFLX.csv')

#Getting a feel of the data available in the data set
# print(df.shape)
# print(df.describe())
# print(df.head())

#Visualize some columns
# plt.figure(figsize=(16,8))
# plt.title('Netflix Stock')
# plt.xlabel('Days')
# plt.ylabel('Close Price USD')
#
# plt.plot(df['Close'])
# plt.show()


df = df[['Close']]
df = df[800:]

# # print(df.head(5))
#
# #Create  variable to predict some 'X' days into the future
future_days = 25
#
# #Create a new column 'Target' shifted 'X' days up
df['Prediction'] = df[['Close']].shift(-future_days)
# print(df.tail())

#Create a feature Data set(X) and convert it to a numpy array and remove last 'x' rows/days
X = np.array(df.drop(['Prediction'],1))[:-future_days]
# print(X)

#Create a target data set (Y) and convert it to a numpy array and get all of the target value expect the last 'x' rows
Y = np.array(df['Prediction'])[:-future_days]
# print(Y)


#Split the data to train and test
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)

#Create the Models

#Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train,y_train)

#Create the Linear Regression model
lr = LinearRegression().fit(x_train,y_train)

#Get the last x rows of the feature data set
x_future = df.drop(['Prediction'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
# print(x_future)

#Show Model tree prediction
tree_prediciton = tree.predict(x_future)
# print(tree_prediciton)
# print()

#Show Model Linear regression
lr_prediction = lr.predict(x_future)
# print(lr_prediction)

#Visualize the data ( Decision Tree Model )

predictions = tree_prediciton
valid = df[X.shape[0]:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model Decision Tree')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','Val','Pred'])
plt.show()


#Visualize the data ( Linear Regression Model )

predictions = lr_prediction
valid = df[X.shape[0]:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model Linear Regression')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','Val','Pred'])
plt.show()
