# RandomForestExample
Random Forest on 30 M records

Task- Predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. 

Following are the steps to make good model using Random Forest Regressor in Python:

1> Loading the data set of test and train. As there are 55M records; we will work on 30M records in order to avoid giving load to the python kernel.

2> Exploratory Data Analysis (EDA): This will help us to know the correlation between the independent variables and also with the target variable. We will also watch for the ouliers if any using EDA.

3> Feature Engineering: This step will helps us to convert the data types of variables (Here Pickup_datetime) and also splitting them into month, day, hour, weekdayname, weekday and year. We will add Haversine distance formula to calculate distance from the given pickup and dropoff latitudes and longitudes.

4> Model Training: We will train using Random Forest Regressor. 

5> Prediction
