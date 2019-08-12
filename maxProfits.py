import pandas as pd
import numpy as np

def prepare_data(company, pathToStockValues):
    '''
    Convert the ‘timestamp’ column to datetime and sort data from
    the CSV file by date by ascending order.
    The CSV must contain a ‘timestamp’ column (YYYY-MM-DD HH:MM:SS) and a ‘close’ column
    '''

    df = pd.read_csv(pathToStockValues + '/' + company + '.csv')
    df['timestamp'] = pd.to_datetime(df.timestamp, format='%Y-%m-%d %H:%M:%S')
    df.index = df['timestamp']
    data = df.sort_index(ascending=True, axis=0)

    new_data = pd.DataFrame(index=range(0,len(df)),columns=['timestamp', 'close'])
    new_data['timestamp'] = data['timestamp'].values
    new_data['close'] = data['close'].values

    return new_data

# def maxProfitsPeriod(companies, pathToStockValues, begin, end):
#     # comapnies : array of symbols
#     # pathToStockValues : string leading to csv files
#     # begin : Timestamp of first day
#     # end : Timestamp of last day
#
#     day = begin
#     returnString = ""
#
#     while (day <= end):
#         returnString += day.strftime("%A %d. %B %Y") + "</br></br>"
#         if (day.weekday() != 5 and day.weekday() != 6):
#             returnString += maxProfitsList(companies, pathToStockValues, day) + "</br>"
#         else:
#             returnString += "Markets were closed</br></br>"
#         #returnString += "----------------------------------------------"
#         day = day + pd.Timedelta(days=1)
#
#     return returnString


def maxProfitsList(companies, pathToStockValues, date):
    # comapnies : array of symbols
    # pathToStockValues : string leading to csv files
    # date : Timestamp of search

    returnList = []

    if (date.weekday() != 5 and date.weekday() != 6):
        prfts = getMaxProfits(companies, pathToStockValues, date)
        for i in range(10):
            #"{} bought at {} and sold at {} would have made a {}% profit<br/>".
            returnList.append([prfts[i][0], prfts[i][1].strftime("%I:%M%p"), prfts[i][2].strftime("%I:%M%p"), round(prfts[i][3]*100, 2)])
    else:
        returnList = None

    return returnList


def getMaxProfits(companies, pathToStockValues, date):
    # comapnies : array of symbols
    # pathToStockValues : string leading to csv files
    # date : Timestamp of search

    profits = []

    for company in companies:
        profits.append(getIndividualMaxProfit(company, pathToStockValues, date))

    return sorted(profits, key=lambda array: array[3], reverse=True)

def getIndividualMaxProfit(company, pathToStockValues, day):
    data = prepare_data(company, pathToStockValues)

    start = 0
    stop = 0

    while(start+1<len(data) and data['timestamp'][start].date() < day.date()):
        start = start+1

    stop=start
    while(stop+1<len(data) and data['timestamp'][stop].date() == day.date()):
        stop = stop+1

    running_min = pd.DataFrame(columns=['minpriceuptillnow', 'timeofthatprice'], index=range(start,stop))

    running_min['minpriceuptillnow'][start] = data['close'][start]
    running_min['timeofthatprice'][start] = start

    if start == stop:
        return company, 0, 0, 0

    for i in range(1, stop-start):
        if(data['close'][start+i] < running_min['minpriceuptillnow'][start+i-1]):
            running_min['minpriceuptillnow'][start+i] = data['close'][start+i]
            running_min['timeofthatprice'][start+i] = start+i
        else:
            running_min['minpriceuptillnow'][start+i] = running_min['minpriceuptillnow'][start+i-1]
            running_min['timeofthatprice'][start+i] = running_min['timeofthatprice'][start+i-1]
        #running_min['minpriceuptillnow'][start+i] = data['close'][start:start+i+1].min()
        #running_min['timeofthatprice'][start+i] = data['close'][start:start+i+1].idxmin()


    substract = pd.DataFrame(index=range(start, stop))
    subtract = data['close'][start:stop] - running_min['minpriceuptillnow']

    sold_time = subtract.values.argmax()
    occurences = np.where(subtract[0:sold_time] == subtract[0:sold_time].min())
    purchase_time = occurences[0][-1]

    profit = (data['close'][start+sold_time]-data['close'][start+purchase_time]) / data['close'][start+purchase_time]

    return company, data['timestamp'][start+purchase_time], data['timestamp'][start+sold_time], profit
