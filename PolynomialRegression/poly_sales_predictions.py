import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from joblib import dump, load

# DataFrame
df = pd.read_csv('datasets/sales/Advertising.csv', index_col=0)

# Preprocessing Data
X = df.drop(['sales'], axis=1)
y = df['sales']

# Creating loop to find the best degree for polynomial dataset.
train_rmse_errors = []
test_rmse_errors = []

for deg in range(1, 6):

    # Convert X dataset to polynomial and degree equals deg
    polynomial_converter = PolynomialFeatures(degree=deg, include_bias=False)
    poly_features = polynomial_converter.fit_transform(X)

    # Train_Test_Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.33, random_state=42)

    # Fit LinearRegression Model
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict Train and Test Set
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    # Calculate errors on Train Set and Test Set
    train_RMSE = np.sqrt(mean_squared_error(y_train, train_prediction))
    test_RMSE = np.sqrt(mean_squared_error(y_test, test_prediction))

    # Append errors to lists for plotting later
    train_rmse_errors.append(train_RMSE)
    test_rmse_errors.append(test_RMSE)

# Plot degrees to see if polynomial over fits
plt.plot(range(1, 6), train_rmse_errors[:5], label='TRAIN')
plt.plot(range(1, 6), test_rmse_errors[:5], label='TEST')
plt.xlabel("Polynomial Complexity")
plt.ylabel("RMSE")
plt.legend()
plt.show()

# Create Final Model When Satisfied With Results
final_poly_converter = PolynomialFeatures(degree=3, include_bias=False)  # best degree was 3
final_poly_features = final_poly_converter.fit_transform(X)

# Fit Final LinearRegression Model
final_model = LinearRegression(n_jobs=-1)
final_model.fit(final_poly_features, y)

# Dump Final Model & Load Final Model
dump(final_model, 'final_poly_model.joblib')
dump(final_poly_converter, 'final_poly_converter.joblib')
load_model = load('final_poly_model.joblib')
loaded_poly = load('final_poly_converter.joblib')

# Use Final Model To Predict Future Sales
campaign = [[250, 97, 88]]  # Campaign equals Advertisement Spent
campaign_poly = loaded_poly.transform(campaign)
final_prediction = load_model.predict(campaign_poly)
print('final prediction: ', final_prediction[0])  # Print Predicted Sales
