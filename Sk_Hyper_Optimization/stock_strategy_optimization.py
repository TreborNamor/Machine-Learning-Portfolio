"""
    The main focus of this project is to hyper optimize a stock trading strategy.
    Skopt will search through a range of parameters and backtest those values
    on a specific dataframe to find which value will create the most profit.
"""

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from skopt.space import Integer
from skopt import Optimizer


# DataFrame
DataFrame = pd.read_csv('datasets/BATS SPY, 1D.csv')

# Buy and sell signals will be appended here
buy_signals = list()
sell_signals = list()

# Skopt will optimize through this range of integers
params_grid = {
    'buy_rsi': Integer(low=1, high=50, name='buy_rsi_value'),
    'sell_rsi': Integer(low=1, high=50, name='buy_rsi_value'),
}


def populate_entry_trend(dataframe: DataFrame) -> DataFrame:
    """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :return: DataFrame with buy column
    """
    dataframe.loc[
        (
            (dataframe['RSI'] <= 30)
            # (dataframe['Histogram'] > 0)
        ),
        'enter_trade'] = 1

    return dataframe


def populate_exit_trend(dataframe: DataFrame) -> DataFrame:
    """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :return: DataFrame with sell column
    """
    dataframe.loc[
        (
            (dataframe['RSI'] >= 70)
            # (dataframe['Histogram'] > 0)
        ),

        'exit_trade'] = 1

    return dataframe


def filter_trades(dataframe: DataFrame) -> DataFrame:
    """
        This method is used to trim data inside our dataframe.
        We start by removing data past our last sell signal,
        then we remove data before our first buy signal.
        This way, our dataframe will start from our first buy signal,
        and end on our last sell signal.
    """
    # Define the data of the first buy signal, and last sell signal
    start_date = dataframe[dataframe['enter_trade'] == 1].index[0]
    end_date = dataframe[dataframe['enter_trade'] == 1].index[-1] + 1

    # Drop data before first buy signal, and data after the last sell signal
    dataframe.drop(dataframe.index[end_date:], axis=0, inplace=True)
    dataframe.drop(dataframe.index[:start_date], axis=0, inplace=True)

    # Drop NaN data between buy and sell signals
    dataframe.dropna(subset=['enter_trade', 'exit_trade'], inplace=True, thresh=1)


def process_trades(dataframe: DataFrame) -> DataFrame:
    """
        This method is used to process buy and sell signals.
        What we want to acheive is a complete open trade to close trade.
        Once a trade is open, future buy signals will be ignored until trade gets a close signal.
        Indicating one full trade session.
        Then script will search for a new position to open a trade.
    """
    # Created iteration variable so script can for loop and properly index dataset
    iteration = dataframe.iterrows()

    # The first iteration to search buy signals.
    for buy_index, buy_trigger in iteration:
        # If buy trigger == 1, enter trade.
        if buy_trigger['enter_trade'] == 1:
            buy_signals.append(buy_trigger['close'])

            # Buy signal found, now search for sell signal
            for sell_index, sell_trigger in iteration:  # might be a better way than iterating two for loops?
                # If sell trigger == 1, exit trade.
                if sell_index > buy_index and sell_trigger['exit_trade'] == 1:
                    sell_signals.append(sell_trigger['close'])
                    break  # we break loop so script can search for the next buy trigger.

    # If there are open trades, we can properly remove them.
    open_trades = len(buy_signals) > len(sell_signals)
    if open_trades:
        buy_signals.pop(-1)


def backtesting(dataframe: DataFrame) -> DataFrame:
    """
        This function will hyperoptimize dataset, and backtest the trading strategies results.
        We are looking for the best parameter values to produce the most profitable strategy.
    """
    pass


populate_entry_trend(dataframe=DataFrame)
populate_exit_trend(dataframe=DataFrame)
filter_trades(dataframe=DataFrame)
process_trades(dataframe=DataFrame)
backtesting(dataframe=DataFrame)
