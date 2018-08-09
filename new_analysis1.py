# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:34:13 2018

@author: adabhoiwala
"""

#--- importing required libraries-----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib
import seaborn as sns

#---- loading data----------
#----- loading 30M records out of 55M records
train = pd.read_csv('C:/Users/Aafreen Dabhoiwala/Documents/Kaggle/NYC/train.csv', nrows=3000000)
test = pd.read_csv('C:/Users/Aafreen Dabhoiwala/Documents/Kaggle/NYC/test.csv')

#-----------------EDA (Exploratory Data Analysis) ------------------------
#--- checking the data types of the variables------
train.dtypes
test.dtypes

#----checking for null values----------
print(train.isnull().sum())
print(test.isnull().sum())

#----- Feature Engineering------------------------------

#----- finding the distance from longtitude and latitude by using Haversine Formula-----
def distance(latitude1, longitude1, latitude2, longitude2):
    data = [train, test]
    for d in data:
        Radius = 6371 # in Kilomoeters
        phi1 = np.radians(d[latitude1])
        phi2 = np.radians(d[latitude2])
    
        delta_phi = np.radians(d[latitude2]-d[latitude1])
        delta_lambda = np.radians(d[longitude2]-d[longitude1])
        
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance = (Radius * c)
        d['distance'] = distance
    return d

distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')

#---- Again checking the number of null values-----
print(train.isnull().sum())
print(test.isnull().sum())

#----- Dropping the null values as their amount are very less------
print('old size of the dataset: %d' %len(train)) 
train = train.dropna(how='any', axis='rows')
print('new size of the dataset: %d' %len(train))

#---- filtering the records as per NYC's latitude and longitude. 
#-- Latitude and Longitude of NYC is 40.730610, -73.935242.
print('old size of the dataset: %d' %len(train))
train2 = train[(train.pickup_latitude>35) & (train.pickup_latitude<45)]
train2 = train2[(train.pickup_longitude>-80) & (train.pickup_longitude<-70)]
train2 = train2[(train.dropoff_latitude>35) & (train.dropoff_latitude<45)]
train2 = train2[(train.dropoff_longitude>-80) & (train.dropoff_longitude<-70)]
print('new size of the dataset: %d' %len(train2))

#----- Removing the negative fare amount from the dataset--------
print('old size of the dataset: %d' %len(train2))
train2 = train2[(train2.fare_amount)>=0]
print('new size of the dataset: %d' %len(train2))

#----- Removing the records with more than 6 passenger_count, as a taxi can accomodate
# not more than 6 passengers
print('old size of the dataset: %d' %len(train2))
train2 = train2[(train2.passenger_count>0) & (train2.passenger_count<=6)]
print('new size of the dataset: %d' %len(train2))

#---- changing the datatype to datetime of Key and Pickup_Datetime Varible to maintain consistency
train2['key']= pd.to_datetime(train2['key'])
train2['pickup_datetime']= pd.to_datetime(train2['pickup_datetime'])
train2.dtypes

test['key'] = pd.to_datetime(test['key'])
test['pickup_datetime']= pd.to_datetime(test['pickup_datetime'])
test.dtypes

# separting pickup_datetime variable into Month, year, WeekDay and hour
train2['pickup_datetime_month'] = train2['pickup_datetime'].dt.month
train2['pickup_datetime_year'] = train2['pickup_datetime'].dt.year
train2['pickup_datetime_day_of_week'] = train2['pickup_datetime'].dt.weekday
train2['pickup_datetime_day_of_hour'] = train2['pickup_datetime'].dt.hour

test['pickup_datetime_month'] = test['pickup_datetime'].dt.month
test['pickup_datetime_year'] = test['pickup_datetime'].dt.year
test['pickup_datetime_day_of_week'] = test['pickup_datetime'].dt.weekday
test['pickup_datetime_day_of_hour'] = test['pickup_datetime'].dt.hour

#-- removing the variables Key and Pickup_Datetime as we will now use the above variables
#of month, year, weekday, hour from the pickup_datetime
train3 = train2.drop("key", axis=1)
train3 = train3.drop("pickup_datetime", axis=1)

test2 = test.drop("key", axis=1)
test2 = test2.drop("pickup_datetime", axis=1)

#------- checking relationship between passengers_count and fare (Scatter plot)-----

plt.figure(figsize=(15,7))
plt.scatter(x=train3['passenger_count'], y=train3['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')

#---- from the above analysis, it shows that single passengers have spend most of the fare;
#--- and they are most frequent travellers


#---- checking the realationship between pickup hours and fare

plt.figure(figsize=(15,7))
plt.scatter(x=train3['pickup_datetime_day_of_hour'], y=train3['fare_amount'], s=1.5)
plt.xlabel('pickup_hour')
plt.ylabel('Fare')

#---- from the above analysis, it shows that fare are comparitively more between 5 AM to 10 AM

#---- chekcing the relationship between pickup day and fare

plt.figure(figsize=(15,7))
plt.scatter(x=train3['pickup_datetime_day_of_week'], y=train3['fare_amount'], s=1.5)
plt.xlabel('pickup_datetime_day_of_week')
plt.ylabel('Fare')

#---- from the above analysis, day of the week doesn't seem to influence the car fare

#------ Checking the relationship between pickup and dropoff latitude and longitude with fares

plt.figure(figsize=(15,7))
plt.scatter(x=train3['pickup_latitude'], y=train3['fare_amount'], s=1.5)
plt.xlabel('pickup_latitude')
plt.ylabel('Fare')

#----- the above analysis shows that, pickup latitude varies with the increase in fare_amount

plt.figure(figsize=(15,7))
plt.scatter(x=train3['pickup_longitude'], y=train3['fare_amount'], s=1.5)
plt.xlabel('pickup_longitude')
plt.ylabel('Fare')


#---------------------MODEL BUILDING-----------------------------

#---- Removing the target variable from the x_train dataset
train4= train3.drop("fare_amount", axis=1)
x_train = train4

#--- adding the target variable to the y_train dataset
y_train = train3['fare_amount'].values

#--- assigning test dataset to x_test
x_test  = test2

#-------- Random Forests-----------------------

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)

"""
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(x_train,y_train)
xgb_predict = xgb.predict(x_test)

""""

#--- TESTING the Prediction--------
submission = pd.read_csv('C:/Users/sample_submission.csv')
submission['fare_amount']= rf_predict
submission.to_csv('C:/Users/new_submission.csv', index=False)

submission.head(20)

#--- with this RMSE is 3.2034----------------------------