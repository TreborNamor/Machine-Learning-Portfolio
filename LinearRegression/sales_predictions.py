from math import sqrt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load

# DataFrame
df = pd.read_csv('datasets/sales/Advertising.csv', index_col=0)

# Preprocessing Data
X = df.drop(['sales'], axis=1)
y = df['sales']

# Train_Test_Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit LinearRegression Model
model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)

# Create List, Model Predictions & Store RSME Into List
test_rsme_error = list()
test_predictions = model.predict(X_test)
rsme = sqrt(mean_squared_error(y_test, test_predictions))
test_rsme_error.append(rsme)
print('test_rsme_error: ', test_rsme_error[0])

# Test Residual To See If LinearRegression Is A Good Fit
test_residuals = (y_test - test_predictions)
sns.scatterplot(x=y_test, y=test_residuals)
plt.axhline(y=0, color='red', ls='--')
sns.displot(test_residuals, bins=25, kde=True)

# Create Final Model When Satisfied With Results
final_model = LinearRegression()
final_model.fit(X, y)
y_hat = final_model.predict(X)

# Plot Actual Sales & Predicted Sales
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
axes[0].plot(df['TV'], df['sales'], 'o')
axes[0].plot(df['TV'], y_hat, 'o', color='red')
axes[0].set_ylabel("Sales")
axes[0].set_title('TV Spend')

axes[1].plot(df['radio'], df['sales'], 'o')
axes[1].plot(df['radio'], y_hat, 'o', color='red')
axes[1].set_ylabel("Sales")
axes[1].set_title('Radio Spend')

axes[2].plot(df['newspaper'], df['sales'], 'o')
axes[2].plot(df['newspaper'], y_hat, 'o', color='red')
axes[2].set_ylabel("Sales")
axes[2].set_title('Newspaper Spend')
plt.tight_layout()
plt.show()

# Dump Final Model & Load Final Model
dump(final_model, 'final_model.joblib')
load_model = load('final_model.joblib')

# Use Final Model To Predict Future Sales compared to
campaign = [[50, 37, 18]]  # Campaign equals Advertisement Spent
final_prediction = load_model.predict(campaign)
print('Final_Prediciton', final_prediction[0])  # Print Predicted Sales
