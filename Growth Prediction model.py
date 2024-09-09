# A simple predictive model that forecasts business growth based on initial capital, market conditions
# and mentorship quality
# Importing the necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt
#creating a sample dataset
data = {'initial_capital': [10000, 15000, 8000, 12000, 20000, 5000,
   18000, 25000, 22000, 30000],
   'market_conditions': [8, 6, 7, 5, 9, 4, 8, 10, 7, 9], #scale of 1-10
   'mentorship_quality': [7, 9, 6, 8, 10, 5, 8, 9, 7, 10], #scale of 1-10
   'business_growth': [15, 22, 12, 18, 30, 9, 25, 35, 28, 40]} # % growth
df = pd.DataFrame(data)
#Features (independent variables) and target (dependent variable)
X = df[['initial_capital' , 'market_conditions' , 'mentorship_quality']]
y = df['business_growth']
#Split the data into testing and training sets(80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Initialize the linear regression model
model = LinearRegression()
#Train the model using the training data
model.fit(X_train, y_train)
#predict using the testing data
y_pred = model.predict(X_test)
#Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#Example new client data
new_client = pd.DataFrame ({
    'initial_capital': [12000],
    'market_conditions': [7],
    'mentorship_quality':[8]
})
#Predict business growth for the new client
predicted_growth = model.predict(new_client)
model.predict(new_client)
print (f"Predicted Business Growth: {predicted_growth[0]}%")

#Plotting the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Growth')
plt.ylabel('Predicted Growth')
plt.title('Actual vs Predicted Business Growth')
plt.show()