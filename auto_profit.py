import methods.predict as prd
import numpy as np


def alternate_buy_sell(data, startAt, stopAt=None):
    """
    Buy, sell, buy, sell etc…
    No predictions are made here, juste calculating a random profit

    Parameters:
        data (pandas.DataFrame): The formated data (see prepare_data
                                 from methods.predict)
        startAt (int): Index where the calulation should start
        stopAt (int): Index where the calulation should stop
            (default is None)

    Return:
        sum (list): Money owned against time
    """
    if stopAt is None:
        stopAt = len(data)

    in_bank = 0
    nb_in_stocks = 1
    sum = [in_bank + nb_in_stocks * data['Close'][startAt-1]]

    for i in range(stopAt-startAt):
        if i%2==0: # we sell first
            in_bank = in_bank + nb_in_stocks * data['Close'][startAt + i]
            nb_in_stocks = 0
        if i%2!=0: #then we buy 1
            in_bank = in_bank - 1 * data['Close'][startAt + i]
            nb_in_stocks = 1
        sum.append(in_bank + nb_in_stocks * data['Close'][startAt + i])

    print('Initial total: {}\n - Stocks: {}\n  - Bank: {}\n\nFinal total: '
    '{}\n - Stocks: {}\n - Bank {}'
    .format(data['Close'][startAt-1], data['Close'][startAt-1], 0,
    in_bank + nb_in_stocks*data['Close'][stopAt-1], nb_in_stocks*data['Close'][stopAt-1], in_bank))

    return sum


def auto(data, methods, thresholds, startAt, stopAt=None):
    """
    Automatic profit maker, make it self the decision to buy, to sell
    or to no nothing given some parameters.

    Parameters:
        data (pandas.DataFrame): The formated data (see prepare_data
                                 from methods.predict)
        methods (list): A list containing the forecasting functions used
                        to predict values (from the methods.predict package)
        thresholds (list): A list containing:
                            - The buy threshold in percent (eg: 0.001 for 0.1%)
                            - The sell threshold
        startAt (int): Index where the calulation should start
        stopAt (int): Index where the calulation should stop
            (default is None)

    Return:
        sum (list): Money owned against time
    """
    if stopAt is None:
        stopAt = len(data)

    buy = thresholds[0]
    sell = thresholds[1]

    diff_data = prd.differentiate(data, 'return-price')

    weights = prd.get_weights(data, methods, stopAt=stopAt, iterations=10, periods=10, diff_order='return-price')

    in_bank = 0
    nb_in_stocks = 1
    sum = [in_bank + nb_in_stocks * data['Close'][startAt-1]]

    for i in range(stopAt-startAt):
        # make prediction on next profit
        pred = np.zeros(len(methods))
        for i, method in enumerate(methods):
            pred[i] = method(diff_data, startAt, startAt+2)[0] * weights[method.__name__]
        prediction = pred.sum()

        current_close = data['Close'][startAt + i]

        if prediction>0 and prediction>buy: # we buy what we can afford
            nb_stocks = nb_stocks + in_bank/current_close
            in_bank = in_bank - in_bank/current_close

        if prediction<0 and abs(prediction)>sell: # we sell all
            in_bank = in_bank + nb_stocks * current_close
            nb_stocks = 0

        sum.append(in_bank + nb_in_stocks * data['Close'][startAt + i])

    print('Initial total: {}\n - Stocks: {}\n  - Bank: {}\n\nFinal total: '
    '{}\n - Stocks: {}\n - Bank {}'
    .format(data['Close'][startAt-1], data['Close'][startAt-1], 0,
    in_bank + nb_in_stocks*data['Close'][stopAt-1], nb_in_stocks*data['Close'][stopAt-1], in_bank))

    return sum
