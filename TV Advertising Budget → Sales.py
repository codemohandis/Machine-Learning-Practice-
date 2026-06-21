import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("advertising.csv")
print(df.head(5))

# Independent variable is "TV" and Dependent variable is "Sales"
x = df[['TV']]
y = df['Sales']

# Visualize Data
plt.xlabel('TV',color = "red")
plt.ylabel('Sales', color = 'red')
plt.scatter(x,y)

# Simple Linear Regression)
lr = linear_model.LinearRegression()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

# Fit the model on the training data
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model
r_squared = lr.score(X_test, y_test)
print(f"\nR-squared on test set: {r_squared:.2f}")

# Predict a new value (example)
new_prediction = lr.predict(pd.DataFrame([[150.5]], columns=['TV']))
print(f"\nPrediction for 150 On TV Ads: {new_prediction[0]:.2f}")


# Visualize Data using Matplotlib
plt.figure(figsize=(10, 6))
plt.xlabel("Tv", color="red")
plt.ylabel("Sales", color="red")
plt.title("Sales Performance: TV Advertising vs. Sales")

# Plot actual test data points
plt.scatter(X_test, y_test, color='blue', label='Actual Test Data')

# Plot the regression line based on the model trained on training data
# Use the full range of X values for plotting the line for better visualization
plt.plot(x, lr.predict(x), color='red', label='Regression Line')
plt.legend()
plt.grid(True)
plt.show()