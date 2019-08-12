import lstm
import boost
import RandomForest as RF
import methods.predict as prd

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
	print(" 2. Moving Average")
	print(" 3. Linear Regression*")
	print(" 4. knn")
	print(" 5. Prophet*")
	print(" 6. Auto-Arima")
	print(" 7. MLP*")
	print(" 8. SVM*")
	print(" 9. Theta")
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
		switch(choixMethode) {
          case 2: predictions = prd.moving_average(diff_data, startAt); break;
		  case 3: predictions = prd.linear_regression(diff_data, startAt); break;
		  case 4: predictions = prd.knn(diff_data, startAt); break;
		  case 5: predictions = prd.prophet(diff_data, startAt); break;
		  case 6: predictions = prd.arima_auto(diff_data, startAt); break;
		  case 7: predictions = prd.MLPRegression(diff_data, startAt, hidden_layer_sizes=tuple([120]*10), activation='relu', solver='lbfgs', batch_size=350); break;
		  case 8: predictions = prd.svm(diff_data, startAt); break;
		  case 9: predictions = predictions = prd.theta_method(diff_data, startAt, alpha=0.12)
		                        prd.plot_predictions(data, predictions, startAt, diff_order=diff_order, print_rms=True)
								break
		  default: predictions = prd.moving_average(diff_data, startAt); break;
		}
		# if (choixMethode == "2"):
		# 	predictions = prd.moving_average(diff_data, startAt)
		# if (choixMethode == "3"):
		# 	predictions = prd.linear_regression(diff_data, startAt)
		# if (choixMethode == "4"):
		# 	predictions = prd.knn(diff_data, startAt)
		# if (choixMethode == "5"):
		# 	predictions = prd.prophet(diff_data, startAt)
		# if (choixMethode == "6"):
		# 	predictions = prd.arima_auto(diff_data, startAt)
		# if (choixMethode == "7"):
		# 	predictions = prd.MLPRegression(diff_data, startAt, hidden_layer_sizes=tuple([120]*10), activation='relu', solver='lbfgs', batch_size=350)
		# if (choixMethode == "8"):
		# 	predictions = prd.svm(diff_data, startAt)
		# if (choixMethode == "9"):
		# 	predictions = prd.theta_method(diff_data, startAt, alpha=0.12)
		# 	prd.plot_predictions(data, predictions, startAt, diff_order=diff_order, print_rms=True)
		prix = data['Close']
		closing_price = prd.inv_differenciate(predictions, diff_order, data['Close'][startAt-1])
		lenTrain = startAt

	profit.periodecomplete(lenTrain, prix, closing_price)
