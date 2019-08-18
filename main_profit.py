import methods.lstm as lstm
import methods.boost as boost
import methods.randomForest as RF
import methods.predict as prd

import numpy as np

import profit

print("What kind of methods do you want to work with ?")
print(" 1. Methods that predict from interval to interval")
print(" 2. Methods that directly predict the entire period ")
butPrgm = input("Enter the choice : ")

if (butPrgm == "1" ):
	print("Which method do you want to work with ?")
	print(" 1. LSTM")
	print(" 2. XGBoost")
	choixMethode = input("Enter the choice : ")

	if (choixMethode == "1" ):
		new_data = lstm.data()
		closing_price, lenTrain, train, valid = lstm.predict(new_data)
		prix = []
		lenTrain = lenTrain + 2
		prix = new_data['close']
	if (choixMethode == "2" ):
		new_data, closing_price_tmp, lenTrain = boost.main()
		prix = []
		prix = new_data['adj_close']
		closing_price = []
		for i in range(len(closing_price_tmp['est'])):
			closing_price.append(closing_price_tmp['est'][lenTrain+i])

	profit.jourlejour(lenTrain, prix, closing_price)





if (butPrgm == "2" ):
	print("Which method do you want to work with ?")
	print(" 1. Random Forest")
	print(" 2. moving_average")
	print(" 3. linear_regression")
	print(" 4. knn")
	print(" 5. prophet")
	print(" 6. arima_auto")
	print(" 7. MLPRegression")
	print(" 8. svm")
	print(" 9. theta_method")
	print(" 10. Combine (with weights) several methods")
	choixMethode = input("Enter the choice : ")

	if (choixMethode == "1" ):
		data, predictions, startAt, train, valid = RF.main()
		prix = data['Close']
		closing_price = prd.inv_differenciate(predictions, 1, data['Close'][startAt-1])
		lenTrain = startAt
	else:
		data = prd.prepare_data('AMZN', ['timestamp', 'close'])
		startAt = 950
		diff_order = 1
		diff_data = prd.differentiate(data, diff_order)
		if (choixMethode == "2"):
		 	predictions = prd.moving_average(diff_data, startAt)
		if (choixMethode == "3"):
		 	predictions = prd.linear_regression(diff_data, startAt)
		if (choixMethode == "4"):
		 	predictions = prd.knn(diff_data, startAt)
		if (choixMethode == "5"):
		 	predictions = prd.prophet(diff_data, startAt)
		if (choixMethode == "6"):
		 	predictions = prd.arima_auto(diff_data, startAt)
		if (choixMethode == "7"):
		 	predictions = prd.MLPRegression(diff_data, startAt, hidden_layer_sizes=tuple([120]*10), activation='relu', solver='lbfgs', batch_size=350)
		if (choixMethode == "8"):
		 	predictions = prd.svm(diff_data, startAt)
		if (choixMethode == "9"):
		 	predictions = prd.theta_method(diff_data, startAt, alpha=0.12)
		if (choixMethode == "10"):
			print("Enter the number of the required methods (Random Forest unavailable). To stop enter 0")
			i = ["1"]
			methods = []
			nb_method = 0
			predictions = []
			noms = []
			while i[nb_method] != "0":
				number = input("Number : ")
				i.append(number)
				if (i[nb_method] == "2"):
					methods.append(prd.moving_average)
					predictions.append(prd.moving_average(diff_data, startAt))
					noms.append("moving_average")
				if (i[nb_method] == "3"):
					methods.append(prd.linear_regression)
					predictions.append(prd.linear_regression(diff_data, startAt))
					noms.append("linear_regression")
				if (i[nb_method] == "4"):
					methods.append(prd.knn)
					predictions.append(prd.knn(diff_data, startAt))
					noms.append("knn")
				if (i[nb_method] == "5"):
					methods.append(prd.prophet)
					predictions.append(prd.prophet(diff_data, startAt))
					noms.append("prophet")
				if (i[nb_method] == "6"):
					methods.append(prd.arima_auto)
					predictions.append(prd.arima_auto(diff_data, startAt))
					noms.append("arima_auto")
				if (i[nb_method] == "7"):
					methods.append(prd.MLPRegression)
					predictions.append(prd.MLPRegression(diff_data, startAt, hidden_layer_sizes=tuple([120]*10), activation='relu', solver='lbfgs', batch_size=350))
					noms.append("MLPRegression")
				if (i[nb_method] == "8"):
					methods.append(prd.svm)
					predictions.append(prd.svm(diff_data, startAt))
					noms.append("svm")
				if (i[nb_method] == "9"):
					methods.append(prd.theta_method)
					predictions.append(prd.theta_method(diff_data, startAt, alpha=0.12))
					noms.append("theta_method")


				nb_method = nb_method + 1


			weights = prd.get_weights(data, methods, iterations=10)

			arr = np.asarray(predictions)

			final_prediction = arr[0] * weights.get(noms[0])
			print(weights.get(noms[0]))
			for j in range(1,nb_method-1):
				final_prediction = final_prediction + arr[j] * weights.get(noms[j])
				print(weights.get(noms[j]))

				predictions = final_prediction



		prix = data['Close']
		closing_price = prd.inv_differenciate(predictions, diff_order, data['Close'][startAt-1])
		lenTrain = startAt

	profit.periodecomplete(lenTrain, prix, closing_price)
