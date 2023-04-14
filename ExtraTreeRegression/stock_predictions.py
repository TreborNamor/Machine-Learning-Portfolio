""" This script is an implementation of Extra Trees Regressor model to predict the close prices of a stock using open
prices as a feature. The dataset used is stored in a CSV file and is read using pandas library. The script first
splits the dataset into training and testing sets and then performs hyperparameter tuning using GridSearchCV. The
best model is then trained with the best hyperparameters obtained from GridSearchCV. The script also calculates and
prints evaluation metrics such as RMSE, MAE, and R^2 for the test set. Finally, it creates a candlestick chart using
plotly library to visualize the actual and predicted close prices and saves it as an HTML file. """
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly.graph_objs as go


# Define Function to Print Evaluation Metrics
def print_evaluation_metrics(rmse, mae, r2):
    print('Model Performance Metrics:')
    print('RMSE: {:.2f}'.format(round(rmse, 2)))
    print('MAE: {:.2f}'.format(round(mae, 2)))
    print('R^2: {:.2f}'.format(round(r2, 2)))


if __name__ == "__main__":
    # Read Data
    filename = 'datasets/BATS_SPY, 1D.csv'
    df = pd.read_csv(filename)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index(pd.DatetimeIndex(df['time']), inplace=True)

    # Split Data into Features and Labels
    X = np.array(df[['open']])
    y = df['close']

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define Extra Trees Regressor Model
    model = ExtraTreesRegressor(n_estimators=100, bootstrap=True, n_jobs=-1, max_depth=None, max_features='sqrt',
                                min_samples_leaf=1, min_samples_split=20)

    # Create Parameter Grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'bootstrap': [True, False],
        'max_features': [1.0, 'sqrt', 'log2']
    }

    # Apply GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                               verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get Best Parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Train Extra Trees Regressor Model with Best Parameters
    best_model = ExtraTreesRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                                     min_samples_split=best_params['min_samples_split'],
                                     min_samples_leaf=best_params['min_samples_leaf'],
                                     bootstrap=best_params['bootstrap'], n_jobs=-1)
    best_model.fit(X_train, y_train)

    # Predict Test Set and Calculate Evaluation Metrics
    y_pred = best_model.predict(X_test)

    # Print Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print_evaluation_metrics(rmse, mae, r2)

    # Create Candlestick Chart and Plot Actual and Predicted Values
    candlestick = go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                 name='Candlestick Chart')

    actual_trace = go.Scatter(x=y_test.index, y=y_test, mode='markers', name='Actual Close',
                              marker=dict(color='gray', symbol='circle-open', size=9))

    predicted_trace = go.Scatter(x=y_test.index, y=y_pred, mode='markers', name='Predicted Close',
                                 marker=dict(color='purple', symbol='x'))

    data = [candlestick, actual_trace, predicted_trace]

    layout = go.Layout(title='Actual vs Predicted Close Prices', xaxis=dict(title='Time'),
                       yaxis=dict(title='Close Price'))
    fig = go.Figure(data=data, layout=layout)

    # Save Plot to HTML File
    filename = 'pictures/actual_vs_predicted_close_prices.html'
    fig.write_html(filename)
