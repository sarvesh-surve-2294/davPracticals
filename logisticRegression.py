# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Sample dataset: Hours studied vs. Pass (1) / Fail (0)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)  # Feature
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])                # Target (binary outcome)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print model parameters
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization
X_range = np.linspace(0, 10, 1000).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_range, y_prob, color='red', label='Logistic Curve')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()
