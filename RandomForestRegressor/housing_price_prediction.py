import os
import tarfile
import urllib.request
from math import sqrt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from joblib import dump, load

# Get data from URL link
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("../RandomForestRegressor/datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """This function is called to get recent dataframe of the California housing market."""
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    dataframe = pd.read_csv(csv_path)
    return dataframe


# Load DataFrame
df = load_housing_data()
df.reset_index()

# Preprocessing Data
dummies = pd.get_dummies(df.ocean_proximity)
df = pd.concat([df, dummies], axis='columns')
df.drop(['ocean_proximity'], axis='columns', inplace=True)
df.dropna(inplace=True)

# Plot housing data for visualization
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=df['population'] / 100, label='population',
        figsize=(10, 7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.show()

# Create features and labels
X = df.drop(['median_house_value'], axis=1)
y = df['median_house_value'].copy()

# Train_Test_Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit RandomForestRegressor Model to train dataset
model = RandomForestRegressor(n_jobs=-1, n_estimators=100, criterion='squared_error')  # Best Estimator
model.fit(X_train, y_train)

# Create List, Model Test Predictions & Store RSME Into List
test_rsme_error = list()
test_predicitions = model.predict(X_test)
rsme = sqrt(mean_squared_error(y_test, test_predicitions))
test_rsme_error.append(rsme)
print('test_rsme_error: ', test_rsme_error[0])

# Create Final Model When Satisfied With Results
final_model = RandomForestRegressor(n_jobs=-1, n_estimators=100, criterion='squared_error')
final_model.fit(X, y)

# Dump Final Model & Load Final Model
dump(final_model, '../LinearRegression/final_model.joblib')
load_model = load('../LinearRegression/final_model.joblib')
