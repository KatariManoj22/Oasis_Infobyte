import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load the dataset
data = pd.read_csv("Advertising.csv")

# Step 2: Data Preprocessing
# Drop the "Unnamed: 0" column from the training data
data = data.drop(columns=["Unnamed: 0"])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=["Sales"])
y = data["Sales"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection and Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Step 6: Prediction
# Example prediction for a new data point
new_data = pd.DataFrame({"TV": [200], "Radio": [40], "Newspaper": [20]})

# Use the trained model to make predictions
predicted_sales = model.predict(new_data)
print("Predicted Sales:", predicted_sales)
