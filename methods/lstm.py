import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def data():
	"""Load the data from the .csv, set the index and create the dataframe
	while sorting the data according to time so that it can be used.

	Parameters:
       

	Returns:
		new_data (dataFrame): contains the closing prices of the share and the time
	"""
	#read the file
	df = pd.read_csv('AMZN.csv', sep= ",")
	print(df.head())

	#setting index as date
	df['timestamp'] = pd.to_datetime(df['timestamp'],format='%Y-%m-%d %H:%M:%S')
	data = df

	#creating dataframe
	new_data = pd.DataFrame(index=range(0,len(df)),columns=['timestamp', 'close'])
	for i in range(0,len(data)):
	    new_data['timestamp'][i] = data['timestamp'][len(data)-i-1]
	    new_data['close'][i] = data['close'][len(data)-i-1]
	new_data.drop('timestamp', axis=1, inplace=True)
	return new_data


    
def predict(new_data):
	"""Split data into test training data. 
	Creates and trains the model and then makes the required predictions.

	Parameters:
		new_data (dataFrame): contains the closing prices of the share and the time
       

	Returns:
		closing_price (list): contains the predicted closing prices
		lenTrain (int) : contains the number of training data
		train (list): contains the data used for training
		valid (list) : contains test data (real values of closing prices)
	"""
	#split into train and test sets
	dataset = new_data.values
	lenTrain = 950
	lenValid = len(new_data)-lenTrain
	train = dataset[0:lenTrain,:]
	valid = dataset[lenTrain:,:]

	#converting dataset into x_train and y_train
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = scaler.fit_transform(dataset)
	x_train, y_train = [], []
	for i in range(60,len(train)):
	    x_train.append(scaled_data[i-60:i,0])
	    y_train.append(scaled_data[i,0])
	x_train, y_train = np.array(x_train), np.array(y_train)
	x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

	#create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
	model.add(LSTM(units=50))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

	#predicting 246 values, using past 60 from the train data
	inputs = new_data[len(new_data) - len(valid) - 60:].values
	inputs = inputs.reshape(-1,1)
	inputs  = scaler.transform(inputs)
	X_test = []
	for i in range(60,inputs.shape[0]):
	    X_test.append(inputs[i-60:i,0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
	closing_price = model.predict(X_test)
	closing_price = scaler.inverse_transform(closing_price)
	return closing_price, lenTrain, train, valid

def plot(new_data, lenTrain, closing_price, train, valid):
	"""Plots closing prices over time for training data, test data and predictions.
	Calculates the RMSE

	Parameters:
		new_data (dataFrame): contains the closing prices of the share and the time
		closing_price (list): contains the predicted closing prices
		lenTrain (int) : contains the number of training data
		train (list): contains the data used for training
		valid (list) : contains test data (real values of closing prices)
       

	Returns:

	"""
	#plot
	train = new_data[:lenTrain]
	valid = new_data[lenTrain:]
	valid['Predictions'] = closing_price
	plt.plot(train['close'])
	plt.plot(valid[['close','Predictions']])
	plt.title('LSTM')
	plt.show()
	#rmse
	rms = np.sqrt(np.mean(np.power((np.array(valid['close'])-np.array(valid['Predictions'])),2)))
	print(rms)

	
###################

'''
new_data = data()
closing_price, lenTrain, train, valid = predict(new_data)
plot(new_data, lenTrain, closing_price, train, valid)
'''
