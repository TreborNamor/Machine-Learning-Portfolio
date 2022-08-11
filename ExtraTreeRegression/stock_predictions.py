from math import sqrt
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Preprocessing DataFrame
df = pd.read_csv('datasets/BATS_SPY, 1D.csv')
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index(pd.DatetimeIndex(df['time']), inplace=True)
del df['time']

# Set Features and Labels
X = np.array(df[['open']])
y = df['close']

# Train_Test_Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04, random_state=42, shuffle=False)


# Fit ExtraTreeRegression Model
model = ExtraTreesRegressor(n_estimators=100, bootstrap=False, n_jobs=-1, warm_start=True)
model.fit(X_train, y_train)

# Create List, Model Predictions & Store RSME Into List
test_rsme_error = list()
test_predicitions = model.predict(X_test)
rsme = sqrt(mean_squared_error(y_test, test_predicitions))
test_rsme_error.append(rsme)
print('test_rsme_error: ', test_rsme_error[0])

# Plot Actual Close(gray) & Predicted Close(purple)
historical_close_time = df.index[:len(y_train)]
actual_close_time = df.index[-len(y_test):]
predicted_close_time = df.index[-len(y_test):]
plt.plot(historical_close_time, y_train,  color='blue',)
plt.plot(actual_close_time, y_test,  color='gray')
plt.plot(predicted_close_time, test_predicitions,  color='purple')
plt.scatter(actual_close_time, y_test,  color='gray')
plt.scatter(predicted_close_time, test_predicitions,  color='purple')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(["Historical Close Price", "Actual Close Price", "Predicted Close Price"], loc=2)
plt.tight_layout()
plt.show()
