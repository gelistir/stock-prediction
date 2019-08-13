#!/usr/bin/env python
# coding: utf-8





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import methods.predict as prd




def main():
	data = prd.prepare_data('AMZN', ['timestamp', 'close'])
	#data = prd.prepare_data('EBAY', distant=True, api_key='IEB6OPAXVXKW0L6B')
	data_copy = data.copy()
	startAt = 950
	stopAt = len(data)

	data_copy['Close'] = data_copy['Close'].diff(periods=1)
	data_copy.drop([0], inplace=True)



	train = data_copy[:startAt]
	valid = data_copy[startAt:stopAt]

	x_train = train.drop('Close', axis=1)
	y_train = train['Close']
	x_valid = valid.drop('Close', axis=1)
	y_valid = valid['Close']


	from sklearn.ensemble import RandomForestRegressor
	model = RandomForestRegressor(criterion='mae', max_depth=None, random_state=0, n_estimators=100)
	model.fit(x_train, y_train)
	predictions = model.predict(x_valid)
	return data, predictions, startAt, train, valid

def plot(data, predictions, startAt):
	prd.plot_predictions(data, predictions, startAt, diff_order=1, print_rms=True)

####################
'''
data, predictions, startAt, train, valid = main()
plot(data, predictions, startAt)
'''

