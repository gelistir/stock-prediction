import methods.predict as prd
import numpy as np


def alternate_buy_sell(data, startAt, stopAt=None):
    """
    Buy, sell, buy, sell etc…
    No predictions are made here, juste checking random profit
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
