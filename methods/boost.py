
import numpy as np
import pandas as pd
import math
import matplotlib

from xgboost import XGBRegressor
from datetime import date
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt


def load_data(stock):
	"""Load the data from the .csv and create the dataframe
	while sorting the data according to time so that it can be used.

	Parameters:
		stock (string): name of the file containing the data
       

	Returns:
		df (dataFrame): contains the closing prices of the share and the time
	"""
	df = pd.read_csv(stock, sep = ",")
	df.loc[:, 'Date'] = pd.to_datetime(df['timestamp'],format='%Y-%m-%d %H:%M:%S')
	df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
	df['month'] = df['date'].dt.month
	df.sort_values(by='date', inplace=True, ascending=True)
	df['adj_close'] = df['close']
	print(df.head())
	return df


def mean_and_std(df, col, N):
	"""Calculates the means and the standard deviation of a column of the dataFrame.

	Parameters:
		df (dataFrame): contains the closing prices of the share and the time
		col (string) : name of the column
		N (int): number of data used to calculate
       

	Returns:
		df_out (dataFrame): contains the closing prices of the share and the time with 2 new columns containing the means and the 				standard deviations
	"""
	means = df[col].rolling(window = N, min_periods=1).mean()
	stds = df[col].rolling(window = N, min_periods=1).std()
	# Add one timestep to the predictions
	means = np.concatenate((np.array([np.nan]), np.array(means[:-1])))
	stds = np.concatenate((np.array([np.nan]), np.array(stds[:-1])))
	df_out = df.copy()
	df_out[col + '_mean'] = means
	df_out[col + '_std'] = stds
	return df_out


def scale_row(row, feat_mean, feat_std):
    """Scales the row data.

	Parameters:
		row (dataFrame): contains the closing prices of the share and the time
		feat_mean (string) : mean of the data
		feat_std (int): std of the data
       

	Returns:
		the scaled data
    """
    feat_std = 0.001 if feat_std == 0 else feat_std # To avoid division by zero
    return (row-feat_mean) / feat_std


def modelisation(X_train_scaled, \
                          y_train_scaled, \
                          X_test_scaled, \
                          y_test, \
                          col_mean, \
                          col_std, \
                          seed=100, \
                          n_estimators=100, \
                          max_depth=3, \
                          learning_rate=0.1, \
                          min_child_weight=1, \
                          subsample=1, \
                          colsample_bytree=1, \
                          colsample_bylevel=1, \
                          gamma=0):
    model = XGBRegressor(seed=100,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma)
    """Create the model.

	Parameters:

       

	Returns:
		rmse (string): root mean squared error between the predictions and the right values
		est (list): contains the predictions
    """
    # Train the model
    model.fit(X_train_scaled, y_train_scaled)
    # Get predicted labels and scale back to original range
    est_scaled = model.predict(X_test_scaled)
    est = est_scaled * col_std + col_mean
    rmse = math.sqrt(mean_squared_error(y_test, est))
    return rmse, est


def plot_adjclose_time(df):
	"""Plot the closing price of the share against the time.

	Parameters:
		df (dataFrame): contains the closing prices of the share and the time

	Returns:

	"""

	'''
	ax = df.plot(y='adj_close', style='g-', grid=True)
	ax.set_ylabel("USD")
	plt.show()
	'''

def diff(df):
	"""Calculate between high and low and between open and close.

	Parameters:
		df (dataFrame): contains the closing prices of the share and the time

	Returns:
		df (dataFrame): contains the closing prices of the share and the time AND new columns containing differences

	"""
	# For each day, calculate difference between high and low
	df['diff_hl'] = df['high'] - df['low']
	df.drop(['high', 'low'], axis=1, inplace=True)
	# For each day, calculate difference between open and close
	df['diff_oc'] = df['open'] - df['close']
	df.drop(['open', 'close'], axis=1, inplace=True)
	print(df.head())
	return df


def feature_engineering(df, N):
	"""Add a column 'order_day' to indicate the order of the rows by date.

	Parameters:
		df (dataFrame): contains the closing prices of the share and the time
		N (int): number of data used to calculate

	Returns:
		df (dataFrame): contains the closing prices of the share and the time AND new columns

	"""
	# Add a column 'order_day'
	df['order_day'] = [x for x in list(range(len(df)))]
	merging_keys = ['order_day']
	# List of columns that we will use to create lags
	lag_cols = ['adj_close', 'diff_hl', 'diff_oc', 'volume']
	shift_range = [x+1 for x in range(N)]
	for shift in tqdm_notebook(shift_range):
	    train_shift = df[merging_keys + lag_cols].copy()
	    train_shift['order_day'] = train_shift['order_day'] + shift
	    foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
	    train_shift = train_shift.rename(columns=foo)
	    df = pd.merge(df, train_shift, on=merging_keys, how='left')
	df = df[N:] # contains NaN
	print(df.head())
	return df


def get_mean_std(df, N):
	"""Gets the means and the standard deviation for different columns of the dataFrame.

	Parameters:
		df (dataFrame): contains the closing prices of the share and the time
		N (int): number of data used to calculate
       

	Returns:
		df (dataFrame): contains the closing prices of the share and the time with new columns containing the means and the 				standard deviations
	"""
	cols_list = ["adj_close","diff_hl","diff_oc","volume"]
	for col in cols_list:
	    df = mean_and_std(df, col, N)
	print(df.head())
	return df


def split_scale(cv_proportion, test_proportion, df, N):
	"""Split data into test, validation and training data. 
	Scales the data.

	Parameters:
		cv_proportion (float) : proportion of data to be used as validation data
		test_proportion (float) : proportion of data to be used as test data
		df (dataFrame): contains the closing prices of the share and the time
		N (int): number of data used to calculate
       

	Returns:
		train (list): contains the data used for training
		cv (list): contains the data used for validation
		test (list): contains the data used for test
		train_scaled (list): contains the data used for training but scaled
		X_train_scaled (list)
		X_cv_scaled (list)
		X_train_cv_scaled (list)
		X_sample_scaled (list)
		y_cv (list)
		y_train (list)
		y_train_scaled (list)
		y_train_cv_scaled (list)
		y_sample (list)
		scaler (scaler) : to scale the data
	"""
	# Split into train, validation and test set
	nb_cv = int(cv_proportion*len(df))
	nb_test = int(test_proportion*len(df))
	nb_train = len(df) - nb_cv - nb_test
	train = df[:nb_train]
	cv = df[nb_train:nb_train+nb_cv]
	train_cv = df[:nb_train+nb_cv]
	test = df[nb_train+nb_cv:]
	# Scale the train, validation and test set
	cols_list = ["adj_close","diff_hl","diff_oc","volume"]
	cols_to_scale = ["adj_close"]
	for j in range(1, N+1):
	    cols_to_scale.append("adj_close_lag_"+str(j))
	    cols_to_scale.append("diff_hl_lag_"+str(j))
	    cols_to_scale.append("diff_oc_lag_"+str(j))
	    cols_to_scale.append("volume_lag_"+str(j))
	# Scale for train set
	scaler = StandardScaler()
	train_scaled = scaler.fit_transform(train[cols_to_scale])
	train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
	train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]
	# Scale for train+dev set
	scaler_train_cv = StandardScaler()
	train_cv_scaled = scaler_train_cv.fit_transform(train_cv[cols_to_scale])
	train_cv_scaled = pd.DataFrame(train_cv_scaled, columns=cols_to_scale)
	train_cv_scaled[['date', 'month']] = train_cv.reset_index()[['date', 'month']]
	# Scale for dev set
	cv_scaled = cv[['date']]
	for col in tqdm_notebook(cols_list):
	    feat_list = [col + '_lag_' + str(shift) for shift in range(1, N+1)]
	    temp = cv.apply(lambda row: scale_row(row[feat_list], row[col+'_mean'], row[col+'_std']), axis=1)
	    cv_scaled = pd.concat([cv_scaled, temp], axis=1)
	# Scale for test set
	test_scaled = test[['date']]
	for col in tqdm_notebook(cols_list):
	    feat_list = [col + '_lag_' + str(shift) for shift in range(1, N+1)]
	    temp = test.apply(lambda row: scale_row(row[feat_list], row[col+'_mean'], row[col+'_std']), axis=1)
	    test_scaled = pd.concat([test_scaled, temp], axis=1)
	# Split into X and y
	sp = []
	for j in range(1, N+1):
	    sp.append("adj_close_lag_"+str(j))
	    sp.append("diff_hl_lag_"+str(j))
	    sp.append("diff_oc_lag_"+str(j))
	    sp.append("volume_lag_"+str(j))
	target = "adj_close"
	X_train = train[sp]
	y_train = train[target]
	X_cv = cv[sp]
	y_cv = cv[target]
	X_train_cv = train_cv[sp]
	y_train_cv = train_cv[target]
	X_sample = test[sp]
	y_sample = test[target]
	# Split into X and y (scaled)
	X_train_scaled = train_scaled[sp]
	y_train_scaled = train_scaled[target]
	X_cv_scaled = cv_scaled[sp]
	X_train_cv_scaled = train_cv_scaled[sp]
	y_train_cv_scaled = train_cv_scaled[target]
	X_sample_scaled = test_scaled[sp]
	return train, cv, test, train_scaled, X_train_scaled, X_cv_scaled, X_train_cv_scaled, X_sample_scaled, y_cv, y_train, y_train_scaled, y_train_cv_scaled, y_sample, scaler


def plot_close_vs_time (train, cv, test):
	"""Plot the closing price against the time.

	Parameters:
		train (list): contains the data used for training
		cv (list): contains the data used for validation
		test (list): contains the data used for test

	Returns:

	"""

	'''
	ax = df.plot(y='adj_close', style='g-', grid=True)
	ax.set_ylabel("USD")
	plt.show()
	'''
	'''
	ax = train.plot( y='adj_close', style='g-', grid=True)
	ax = cv.plot( y='adj_close', style='y-', grid=True, ax=ax)
	ax = test.plot( y='adj_close', style='b-', grid=True, ax=ax)
	ax.legend(['train', 'dev', 'test'])
	ax.set_ylabel("USD")
	ax.set_title("Without scaling")
	plt.show()
	'''

def create_model(X_train_scaled, y_train_scaled, model_seed, n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, colsample_bylevel, gamma):
	"""Create the model with the train data and the hyper-parameters found in the others functions

	Parameters:
		X_train_scaled (list) : closing prices for the train splited and scaled in another function
		y_train_scaled (list) : closing prices for the train splited and scaled in another function
		model_seed (int) : hyper-parameter
		n_estimators (int) : Number of different trees to be tested
		max_depth (int) : Maximum tree depth
		learning_rate (float) : Learning rate
		min_child_weight (int) : Minimum number of threads in each node
		subsample (float) : Proportion of sub-sample of each node
		colsample_bytree (float) : proportion of subsample columns when building each tree
		colsample_bylevel (float) : 
		gamma (float) : Minimum loss of RMSE required to perform an additional partition on a node of the tree
       

	Returns:
		model (XGBRegressor): contains the information concerning the created model
	"""
	model = XGBRegressor(seed=model_seed,
		             n_estimators=n_estimators,
		             max_depth=max_depth,
		             learning_rate=learning_rate,
		             min_child_weight=min_child_weight,
		             subsample=subsample,
		             colsample_bytree=colsample_bytree,
		             colsample_bylevel=colsample_bylevel,
		             gamma=gamma)
	# Train the regressor
	model.fit(X_train_scaled, y_train_scaled)
	return model


def predict_train(model, X_train_scaled, y_train, scaler, train, cv, test):
	"""Makes the predictions on the train data. Just to check the quality of the model.

	Parameters:
		model (XGBRegressor): contains the information concerning the created model
		X_train_scaled (list) : closing prices for the train splited and scaled in another function
		y_train (list) : closing prices for the train splited in another function
		train (list): contains the data used for training
		cv (list): contains the data used for validation
		test (list): contains the data used for test
		scaler (scaler): to scale the data
       

	Returns:

	"""
	est_scaled = model.predict(X_train_scaled)
	est = est_scaled * math.sqrt(scaler.var_[0]) + scaler.mean_[0]
	print("RMSE on train set = %0.3f" % math.sqrt(mean_squared_error(y_train, est)))
	'''	
	# Plot
	est_df = pd.DataFrame({'est': est,
		               'date': train['date']})
	ax = train.plot( y='adj_close', style='g-', grid=True)
	ax = cv.plot( y='adj_close', style='y-', grid=True, ax=ax)
	ax = test.plot( y='adj_close', style='b-', grid=True, ax=ax)
	ax = est_df.plot( y='est', style='r-', grid=True, ax=ax)
	ax.legend(['train', 'dev', 'test', 'predictions'])
	ax.set_ylabel("USD")
	ax.set_title('Without scaling')
	plt.show()
	'''


def predict_test(model, y_cv, X_cv_scaled, cv, train, test):
	"""Makes the predictions on the test data.

	Parameters:
		model (XGBRegressor): contains the information concerning the created model
		y_cv (list) : closing prices for the train splited in another function
		X_cv_scaled (list) : closing prices for the train splited and scaled in another function
		train (list): contains the data used for training
		cv (list): contains the data used for validation
		test (list): contains the data used for test
       

	Returns:
		rmse_before (float) : root mean squared error between the predictions and the right values

	"""
	est_scaled = model.predict(X_cv_scaled)
	cv['est_scaled'] = est_scaled
	cv['est'] = cv['est_scaled'] * cv['adj_close_std'] + cv['adj_close_mean']
	rmse_before = math.sqrt(mean_squared_error(y_cv, cv['est']))
	print("RMSE on dev set = %0.3f" % rmse_before)
	'''
	# Plot
	est_df = pd.DataFrame({'est': cv['est'],
		               'y_cv': y_cv,
		               'date': cv['date']})

	ax = train.plot( y='adj_close', style='g-', grid=True)
	ax = cv.plot( y='adj_close', style='y-', grid=True, ax=ax)
	ax = test.plot( y='adj_close', style='b-', grid=True, ax=ax)
	ax = est_df.plot( y='est', style='r-', grid=True, ax=ax)
	ax.legend(['train', 'dev', 'test', 'predictions'])
	ax.set_ylabel("USD")
	plt.show()
	# Plot for dev set only
	ax = train.plot( y='adj_close', style='b-', grid=True)
	ax = cv.plot( y='adj_close', style='y-', grid=True, ax=ax)
	ax = test.plot( y='adj_close', style='g-', grid=True, ax=ax)
	ax = est_df.plot( y='est', style='r-', grid=True, ax=ax)
	ax.legend(['train', 'dev', 'test', 'predictions'])
	ax.set_ylabel("USD")
	'''
	return rmse_before


def XGBoost_n_estimators_max_depth(cv, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv):
	"""Find the best values (which minimise the rmse) of n_estimators and max_depth.

	Parameters:
		cv (list): contains the data used for validation
		X_train_scaled (list)
		y_train_scaled (list)
		X_cv_scaled (list)
		y_cv (list)
       

	Returns:
		n_estimators_opt (int) : best value of the hyper-parameter
		max_depth_opt (int) : best value of the hyper-parameter

	"""
	param_name = 'n_estimators'
	param_ex = range(10, 300, 10)
	param2_name = 'max_depth'
	param2_ex = [2, 3, 4, 5, 6, 7, 8]
	error_rate = {param_name: [] , param2_name: [], 'rmse': []}
	for param in tqdm_notebook(param_ex):
		for param2 in param2_ex:
			rmse, _ = modelisation(X_train_scaled,
		                             y_train_scaled,
		                             X_cv_scaled,
		                             y_cv,
		                             cv['adj_close_mean'],
		                             cv['adj_close_std'],
		                             seed=100,
		                             n_estimators=param,
		                             max_depth=param2,
		                             learning_rate=0.1,
		                             min_child_weight=1,
		                             subsample=1,
		                             colsample_bytree=1,
		                             colsample_bylevel=1,
		                             gamma=0)
			error_rate[param_name].append(param)
			error_rate[param2_name].append(param2)
			error_rate['rmse'].append(rmse)
	error_rate = pd.DataFrame(error_rate)
	'''
	# Plot performance versus params
	temp = error_rate[error_rate[param2_name]==param2_ex[0]]
	ax = temp.plot(x=param_name, y='rmse', style='bo-')
	legend_list = [param2_name + '_' + str(param2_ex[0])]
	color_list = ['r', 'g', 'k', 'y', 'm', 'c', '0.75']
	for i in range(1,len(param2_ex)):
	    temp = error_rate[error_rate[param2_name]==param2_ex[i]]
	    ax = temp.plot(x=param_name, y='rmse', color=color_list[i%len(color_list)], marker='o', ax=ax)
	    legend_list.append(param2_name + '_' + str(param2_ex[i]))
	ax.set_xlabel(param_name)
	ax.set_ylabel("RMSE")
	plt.legend(legend_list, loc='best', bbox_to_anchor=(0.5, 1))
	plt.show()
	'''
	# Get optimum value for param and param2, using RMSE
	temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
	n_estimators_opt = temp['n_estimators'].values[0]
	max_depth_opt = temp['max_depth'].values[0]
	print("min RMSE = %0.3f" % error_rate['rmse'].min())
	print("optimum params = ")
	print(n_estimators_opt, ' ', max_depth_opt)
	return n_estimators_opt, max_depth_opt


def XGBoost_learning_rate_min_child_weight(n_estimators_opt,  max_depth_opt, cv, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv):
	"""Find the best values (which minimise the rmse) of learning_rate and min_child_weight.

	Parameters:
		cv (list): contains the data used for validation
		X_train_scaled (list)
		y_train_scaled (list)
		X_cv_scaled (list)
		y_cv (list)
		n_estimators_opt (int) : best value of the hyper-parameter
		max_depth_opt (int) : best value of the hyper-parameter
       

	Returns:
		learning_rate_opt (float) : best value of the hyper-parameter
		min_child_weight_opt (int) : best value of the hyper-parameter

	"""
	param_name = 'learning_rate'
	param_ex = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
	param2_name = 'min_child_weight'
	param2_ex = range(5, 20, 1)
	error_rate = {param_name: [] , param2_name: [], 'rmse': []}
	for param in tqdm_notebook(param_ex):
		for param2 in param2_ex:
			rmse, _ = modelisation(X_train_scaled,
		                             y_train_scaled,
		                             X_cv_scaled,
		                             y_cv,
		                             cv['adj_close_mean'],
		                             cv['adj_close_std'],
		                             seed=100,
		                             n_estimators=n_estimators_opt,
		                             max_depth=max_depth_opt,
		                             learning_rate=param,
		                             min_child_weight=param2,
		                             subsample=1,
		                             colsample_bytree=1,
		                             colsample_bylevel=1,
		                             gamma=0)
			error_rate[param_name].append(param)
			error_rate[param2_name].append(param2)
			error_rate['rmse'].append(rmse)
	error_rate = pd.DataFrame(error_rate)
	'''
	# Plot performance versus params
	temp = error_rate[error_rate[param2_name]==param2_ex[0]]
	ax = temp.plot(x=param_name, y='rmse', style='bo-')
	legend_list = [param2_name + '_' + str(param2_ex[0])]
	color_list = ['r', 'g', 'k', 'y', 'm', 'c', '0.75']
	for i in range(1,len(param2_ex)):
	    temp = error_rate[error_rate[param2_name]==param2_ex[i]]
	    ax = temp.plot(x=param_name, y='rmse', color=color_list[i%len(color_list)], marker='o', ax=ax)
	    legend_list.append(param2_name + '_' + str(param2_ex[i]))
	ax.set_xlabel(param_name)
	ax.set_ylabel("RMSE")
	plt.legend(legend_list, loc='best', bbox_to_anchor=(0.5, 1))
	plt.show()
	'''
	# Get optimum value for param and param2, using RMSE
	temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
	learning_rate_opt = temp['learning_rate'].values[0]
	min_child_weight_opt = temp['min_child_weight'].values[0]
	print("min RMSE = %0.3f" % error_rate['rmse'].min())
	print("optimum params = ")
	print(learning_rate_opt,' ', min_child_weight_opt)
	return learning_rate_opt, min_child_weight_opt


def XGBoost_subsample_gamma(n_estimators_opt,  max_depth_opt, learning_rate_opt, min_child_weight_opt, cv, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv):
	"""Find the best values (which minimise the rmse) of subsample and gamma.

	Parameters:
		cv (list): contains the data used for validation
		X_train_scaled (list)
		y_train_scaled (list)
		X_cv_scaled (list)
		y_cv (list)
		n_estimators_opt (int) : best value of the hyper-parameter
		max_depth_opt (int) : best value of the hyper-parameter
		learning_rate_opt (float) : best value of the hyper-parameter
		min_child_weight_opt (int) : best value of the hyper-parameter
       

	Returns:
		subsample_opt (float) : best value of the hyper-parameter
		gamma_opt (float) : best value of the hyper-parameter
	"""

	param_name = 'subsample'
	param_ex = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	param2_name = 'gamma'
	param2_ex = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	error_rate = {param_name: [] , param2_name: [], 'rmse': []}
	for param in tqdm_notebook(param_ex):
		for param2 in param2_ex:
			rmse, _ = modelisation(X_train_scaled,
		                             y_train_scaled,
		                             X_cv_scaled,
		                             y_cv,
		                             cv['adj_close_mean'],
		                             cv['adj_close_std'],
		                             seed=100,
		                             n_estimators=n_estimators_opt,
		                             max_depth=max_depth_opt,
		                             learning_rate=learning_rate_opt,
		                             min_child_weight=min_child_weight_opt,
		                             subsample=param,
		                             colsample_bytree=1,
		                             colsample_bylevel=1,
		                             gamma=param2)
			error_rate[param_name].append(param)
			error_rate[param2_name].append(param2)
			error_rate['rmse'].append(rmse)
	error_rate = pd.DataFrame(error_rate)
	'''
	# Plot performance versus params
	temp = error_rate[error_rate[param2_name]==param2_ex[0]]
	ax = temp.plot(x=param_name, y='rmse', style='bo-')
	legend_list = [param2_name + '_' + str(param2_ex[0])]
	color_list = ['r', 'g', 'k', 'y', 'm', 'c', '0.75']
	for i in range(1,len(param2_ex)):
	    temp = error_rate[error_rate[param2_name]==param2_ex[i]]
	    ax = temp.plot(x=param_name, y='rmse', color=color_list[i%len(color_list)], marker='o', ax=ax)
	    legend_list.append(param2_name + '_' + str(param2_ex[i]))
	ax.set_xlabel(param_name)
	ax.set_ylabel("RMSE")
	plt.legend(legend_list, loc='best', bbox_to_anchor=(0.5, 1))
	plt.show()
	'''
	# Get optimum value for param and param2, using RMSE
	temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
	subsample_opt = temp['subsample'].values[0]
	gamma_opt = temp['gamma'].values[0]
	print("min RMSE = %0.3f" % error_rate['rmse'].min())
	print("optimum params = ")
	print(subsample_opt, ' ', gamma_opt)
	return subsample_opt, gamma_opt


def XGBoost_colsample_bytree_colsample_bylevel(n_estimators_opt,  max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, gamma_opt, cv, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv):
	"""Find the best values (which minimise the rmse) of colsample_bytree and colsample_bylevel.

	Parameters:
		cv (list): contains the data used for validation
		X_train_scaled (list)
		y_train_scaled (list)
		X_cv_scaled (list)
		y_cv (list)
		n_estimators_opt (int) : best value of the hyper-parameter
		max_depth_opt (int) : best value of the hyper-parameter
		learning_rate_opt (float) : best value of the hyper-parameter
		min_child_weight_opt (int) : best value of the hyper-parameter
		subsample_opt (float) : best value of the hyper-parameter
		gamma_opt (float) : best value of the hyper-parameter
       

	Returns:
		colsample_bytree_opt (float) : best value of the hyper-parameter
		colsample_bylevel_opt (float) : best value of the hyper-parameter
		error_rate (float) : rmse with the new prediction
	"""
	param_name = 'colsample_bytree'
	param_ex = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
	param2_name = 'colsample_bylevel'
	param2_ex = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
	error_rate = {param_name: [] , param2_name: [], 'rmse': []}
	for param in tqdm_notebook(param_ex):
		for param2 in param2_ex:
			rmse, _ = modelisation(X_train_scaled,
		                             y_train_scaled,
		                             X_cv_scaled,
		                             y_cv,
		                             cv['adj_close_mean'],
		                             cv['adj_close_std'],
		                             seed=100,
		                             n_estimators=n_estimators_opt,
		                             max_depth=max_depth_opt,
		                             learning_rate=learning_rate_opt,
		                             min_child_weight=min_child_weight_opt,
		                             subsample=subsample_opt,
		                             colsample_bytree=param,
		                             colsample_bylevel=param2,
		                             gamma=gamma_opt)
			error_rate[param_name].append(param)
			error_rate[param2_name].append(param2)
			error_rate['rmse'].append(rmse)
	error_rate = pd.DataFrame(error_rate)
	'''
	# Plot performance versus params
	temp = error_rate[error_rate[param2_name]==param2_ex[0]]
	ax = temp.plot(x=param_name, y='rmse', style='bo-')
	legend_list = [param2_name + '_' + str(param2_ex[0])]
	color_list = ['r', 'g', 'k', 'y', 'm', 'c', '0.75']
	for i in range(1,len(param2_ex)):
	    temp = error_rate[error_rate[param2_name]==param2_ex[i]]
	    ax = temp.plot(x=param_name, y='rmse', color=color_list[i%len(color_list)], marker='o', ax=ax)
	    legend_list.append(param2_name + '_' + str(param2_ex[i]))
	ax.set_xlabel(param_name)
	ax.set_ylabel("RMSE")
	plt.legend(legend_list, loc='best', bbox_to_anchor=(0.5, 1))
	plt.show()
	'''
	# Get optimum value for param and param2, using RMSE
	temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
	colsample_bytree_opt = temp['colsample_bytree'].values[0]
	colsample_bylevel_opt = temp['colsample_bylevel'].values[0]
	print("min RMSE = %0.3f" % error_rate['rmse'].min())
	print("optimum params = ")
	print(colsample_bytree_opt, ' ', colsample_bylevel_opt)
	return colsample_bytree_opt, colsample_bylevel_opt, error_rate


def changing(rmse_before, n_estimators_opt,  max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, colsample_bytree_opt, colsample_bylevel_opt, gamma_opt, error_rate):
	"""Show the changes between old and new hyper-parameters and improved rmse.

	Parameters:
		rmse_before (float) : root mean squared error between the old predictions and the right values
		error_rate (float) : rmse with the new prediction
		n_estimators_opt (int) : best value of the hyper-parameter
		max_depth_opt (int) : best value of the hyper-parameter
		learning_rate_opt (float) : best value of the hyper-parameter
		min_child_weight_opt (int) : best value of the hyper-parameter
		subsample_opt (float) : best value of the hyper-parameter
		gamma_opt (float) : best value of the hyper-parameter
       

	Returns:

	"""
	d = {'param': ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight', 'subsample', 'colsample_bytree', 'colsample_bylevel', 'gamma', 'rmse'],
	     'original': [100, 3, 0.1, 1, 1, 1, 1, 0, rmse_before],
	     'after_tuning': [n_estimators_opt, max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, colsample_bytree_opt, colsample_bylevel_opt, gamma_opt, error_rate['rmse'].min()]}
	tuned_params = pd.DataFrame(d)
	tuned_params = tuned_params.round(3)
	print(tuned_params)


def final_model(X_train_cv_scaled, y_train_cv_scaled, X_sample_scaled, y_sample, n_estimators_opt, max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, colsample_bytree_opt, colsample_bylevel_opt, gamma_opt, train, cv, test):
	"""Create the new model (with the best values of the hyper-parameters.

	Parameters:
		X_train_cv_scaled (list)
		y_train_cv_scaled (list)
		X_sample_scaled (list)
		y_sample (list)
		rmse_before (float) : root mean squared error between the old predictions and the right values
		error_rate (float) : rmse with the new prediction
		n_estimators_opt (int) : best value of the hyper-parameter
		max_depth_opt (int) : best value of the hyper-parameter
		learning_rate_opt (float) : best value of the hyper-parameter
		min_child_weight_opt (int) : best value of the hyper-parameter
		subsample_opt (float) : best value of the hyper-parameter
		gamma_opt (float) : best value of the hyper-parameter
		cv (list): contains the data used for validation
		test (list): contains the data used for test
		scaler (scaler): to scale the data
       

	Returns:
		est_df (dataFrame) : contains the predicted values and the time

	"""
	rmse, est = modelisation(X_train_cv_scaled,
		                     y_train_cv_scaled,
		                     X_sample_scaled,
		                     y_sample,
		                     test['adj_close_mean'],
		                     test['adj_close_std'],
		                     seed=100,
		                     n_estimators=n_estimators_opt,
		                     max_depth=max_depth_opt,
		                     learning_rate=learning_rate_opt,
		                     min_child_weight=min_child_weight_opt,
		                     subsample=subsample_opt,
		                     colsample_bytree=colsample_bytree_opt,
		                     colsample_bylevel=colsample_bylevel_opt,
		                     gamma=gamma_opt)
	print("RMSE on test set = %0.3f" % rmse)
	# Plot
	est_df = pd.DataFrame({'est': est, 'y_sample': y_sample, 'date': test['date']})
	'''
	ax = train.plot( y='adj_close', style='g-', grid=True)
	ax = cv.plot( y='adj_close', style='y-', grid=True, ax=ax)
	ax = test.plot( y='adj_close', style='b-', grid=True, ax=ax)
	ax = est_df.plot( y='est', style='r-', grid=True, ax=ax)
	ax.legend(['train', 'dev', 'test', 'predictions'])
	ax.set_ylabel("USD")
	
	plt.show()
	 Plot only for test set
	ax = test.plot( y='adj_close', style='bx-', grid=True)
	ax = est_df.plot( y='est', style='rx-', grid=True, ax=ax)
	ax.legend(['test', 'predictions using xgboost'], loc='upper left')
	ax.set_ylabel("USD")
	plt.show()
	'''

	rms = np.sqrt(np.mean(np.power((np.array(test['adj_close'])-np.array(est_df['est'])),2)))
	print(rms)
	return est_df

#################################################################################
def main():
	"""Main

	Parameters:       

	Returns:
		df (dataFrame): contains the closing prices of the share and the time
		est_df (dataFrame) : contains the predicted values and the time
		lenTrain (int) : contains the number of training data
	"""

	stock = "AMZN.csv"
	test_proportion = 0.15
	cv_proportion = 0.15
	N = 3

	n_estimators = 100
	max_depth = 3
	learning_rate = 0.1
	min_child_weight = 1
	subsample = 1
	colsample_bytree = 1
	colsample_bylevel = 1
	gamma = 0
	model_seed = 100


	df = load_data(stock)

	diff(df)

	df = feature_engineering(df, N)

	df = get_mean_std(df, N)

	train, cv, test, train_scaled, X_train_scaled, X_cv_scaled, X_train_cv_scaled, X_sample_scaled, y_cv, y_train, y_train_scaled, y_train_cv_scaled, y_sample, scaler = split_scale(cv_proportion, test_proportion, df, N)

	plot_close_vs_time (train, cv, test)

	model = create_model(X_train_scaled, y_train_scaled, model_seed, n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, colsample_bylevel, gamma)

	predict_train(model, X_train_scaled, y_train, scaler, train, cv, test)

	rmse_before = predict_test(model, y_cv, X_cv_scaled, cv, train, test)

	n_estimators_opt, max_depth_opt = XGBoost_n_estimators_max_depth(cv, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv)

	learning_rate_opt, min_child_weight_opt = XGBoost_learning_rate_min_child_weight(n_estimators_opt,  max_depth_opt,cv, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv)

	subsample_opt, gamma_opt = XGBoost_subsample_gamma(n_estimators_opt,  max_depth_opt, learning_rate_opt, min_child_weight_opt, cv, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv)

	colsample_bytree_opt, colsample_bylevel_opt, error_rate = XGBoost_colsample_bytree_colsample_bylevel(n_estimators_opt,  max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, gamma_opt, cv, X_train_scaled, y_train_scaled, X_cv_scaled, y_cv)

	changing(rmse_before, n_estimators_opt,  max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, colsample_bytree_opt, colsample_bylevel_opt, gamma_opt, error_rate)

	est_df = final_model(X_train_cv_scaled, y_train_cv_scaled, X_sample_scaled, y_sample, n_estimators_opt, max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, colsample_bytree_opt, colsample_bylevel_opt, gamma_opt, train, cv, test)

	lenTrain = len(train) + len(cv) + 3 # lignes Nan du début qui ne sont pas prises en compte

	return df, est_df, lenTrain

#df, est_df, lenTrain = main()



