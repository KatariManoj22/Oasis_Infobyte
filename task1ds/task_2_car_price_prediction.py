import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load Dataset
data = pd.read_csv('D:/#volume_D/#Internships & CSP/Oasis_Infobyte/car_data.csv')

# Step 2: Data Preprocessing
data.dropna(inplace=True)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']
X = pd.get_dummies(X)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Model
model = XGBRegressor()
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 7: Prediction for new data
new_data = pd.DataFrame({"Car_Name": ["ertiga"],"Year": [2020],"Driven_kms":[50000],"Fuel_Type":["Diesel"]})

# Perform one-hot encoding for categorical variables
new_data_encoded = pd.get_dummies(new_data, columns=["Car_Name", "Fuel_Type"])

# Ensure that the new_data_encoded has the same columns as the training data
missing_cols = set(X_train.columns) - set(new_data_encoded.columns)
for col in missing_cols:
    new_data_encoded[col] = 0

# Reorder columns to match the order of columns in the training data
new_data_encoded = new_data_encoded[X_train.columns]

# Use the trained model to make predictions
predicted_price = model.predict(new_data_encoded)
print("Predicted Price:", predicted_price)
