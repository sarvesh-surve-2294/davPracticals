# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dataset: Hours studied vs. Marks obtained
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)  # Feature (e.g., hours studied)
y = np.array([10, 20, 30, 40, 50, 60, 65, 78, 90])        # Target (e.g., marks)

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print model parameters
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Plot the regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.legend()
plt.grid(True)
plt.show()
