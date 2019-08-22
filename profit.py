
import matplotlib.pyplot as plt
import numpy as np


def jourlejour(lenTrain, prix, closing_price):
	"""Launches the simulation for a method that makes predictions from t to t+1

	Parameters:
		lenTrain (int) : contains the number of training data
		prix (list) : closing prices of the share for the entire period
		closing_price (list) : predicted  closing prices of the share for the test period 
       

	Returns:

	"""
	action_detenu = 0
	argent = 0.0
	histo_argent = [0.0]
	gains = 0.0
	histo_gains = [0.0]
	prix_actuel_tmp = 0.0
	for i in range(0,len(prix)-lenTrain):

		prix_actuel = prix[lenTrain+i]
		print('\nHere is your current portfolio \nMoney :', argent, '\nShares held : ', action_detenu, '(unit cost ', prix_actuel_tmp, ')\nThis means a total gain/loss of : ', gains, '\n')

		c = plt.subplot(1, 2, 1)	
		c.plot(histo_gains, 'g')
		c.set_title('History of our total gains/losses')
		c.set_ylabel('Gains/Losses (in dollars)')
		d = plt.subplot(1, 2, 2)
		d.plot(histo_argent, 'g')
		d.set_title("History of the money in our possession")
		d.set_ylabel('Money (in dollars)')
		plt.show()


		actuel = prix[:lenTrain-2+i]
		last10 = prix[lenTrain+i-12:lenTrain+i-2]

		a = plt.subplot(2, 1, 1)
		a.plot(actuel)
		a.set_title("Closing price of the action as a function of time")
		a.set_ylabel('Closing price (in dollars)')
		b = plt.subplot(2, 1, 2)
		b.plot(last10)
		b.set_title('Focus on the last 10 closing prices')
		b.set_ylabel('Closing price (in dollars)')
		plt.show()


		prix_prevu = closing_price[i+1]
		print("Current share price : ", prix_actuel)
		print('Expected price at the next closing : ', prix_prevu, '\n')

		rep = input('Do you want to buy ? [y/n] ')
		if rep == 'y':
			nb = input('How many do you want to buy ? ')
			action_detenu = action_detenu + int(nb)
			argent = argent - int(nb) * prix_actuel
		print('\n')

		rep = input('Do you want to sell ? [y/n] ')
		if (rep == 'y' and action_detenu > 0):
			nb = input('How many do you want to sell ? ')
			if int(nb) <= action_detenu:
				action_detenu = action_detenu - int(nb)
				argent = argent + int(nb) * prix_actuel
			else:
				print("You can't sell more than you own")
		else:
			if (rep == 'o' and action_detenu <= 0):
				print("Impossible, you don't have any shares")
		print('\n')

		gains = argent+action_detenu*prix_actuel
		histo_gains.append(gains)
		histo_argent.append(argent)
		prix_actuel_tmp = prix_actuel

		rep = input('Do you want to continue ? [y/n] ')
		if rep == 'n':
			break

	print('\nHere is your final portfolio \nMoney :', argent, '\nShares held : ', action_detenu, '(unit cost ', prix_actuel, ')\nThis means a total gain/loss of : ', gains, ' (you started at 0)')
	c = plt.subplot(2, 1, 1)	
	c.plot(histo_gains, 'g')
	c.set_title('History of our total gains/losses')
	c.set_ylabel('Gains/Losses (in dollars)')
	d = plt.subplot(2, 1, 2)
	d.plot(histo_argent, 'g')
	d.set_title("History of the money in our possession")
	d.set_ylabel('Money (in dollars)')
	plt.show()

##########################################################

def periodecomplete(lenTrain, prix, closing_price):
	"""Launches the simulation for a method that makes predictions for the entier period

	Parameters:
		lenTrain (int) : contains the number of training data
		prix (list) : closing prices of the share for the entire period
		closing_price (list) : predicted  closing prices of the share for the test period 
       

	Returns:

	"""
	prix_actuel = prix[lenTrain-1]
	max_prediction = max(closing_price)
	gain = max_prediction - prix_actuel
	ind = np.argmax(closing_price)
	prix_reel = prix[lenTrain+ind]

	actuel = prix[:lenTrain]
	last10 = prix[lenTrain-10:lenTrain]
	a = plt.subplot(2, 1, 1)
	a.plot(actuel)
	a.set_title("Closing price of the action as a function of time")
	a.set_ylabel('Closing price (in dollars)')
	b = plt.subplot(2, 1, 2)
	b.plot(last10)
	b.set_title('Focus on the last 10 closing prices')
	b.set_ylabel('Closing price (in dollars)')
	plt.show()

	print('\nWith the chosen method, it is estimated that the maximum gain per share will be ', gain)
	print("Current price (of possible purchase) : ", prix_actuel, " / Estimated maximum price (of possible sale) : ", max_prediction)
	rep = input('Do you want to take the risk and buy shares ? [y/n] ')
	if rep == 'y':
		nb = input('How many do you want to buy ? ')
		argent = - int(nb) * prix_actuel
		print("\nFinally, the actual share price at the time it was expected to be max is ", prix_reel)
		gain_reel = argent + int(nb) * prix_reel
		print("The real gain is therefore ", gain_reel)
		if gain_reel > 0:
			print("\nCongratulations !")
		else:
			print("\nToo bad !")

	else:
		print("\nEnd of the program, you're not a player !")
	

