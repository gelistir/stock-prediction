# Stock Prediction Package

## Dependencies

All the dependencies of the project are listed in the `requirements.txt` file. To install them, we recommend to use a virtual environment:

```bash
python3 -m venv venv/
source venv/bin/activate
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

To quit the virtual environment just run:

```bash
deactivate
```

## Usage

The main functions are in the `methods/predict.py` file.
In order to use them in a script import the package with:

```python
import methods.predict as prd
```

To retreive data either use the `data` folder or get an [AlphaVantage API key](https://www.alphavantage.co/).
Then, to import the stock dataframe use:

```python
data = prd.prepare_data('data/GOOGL') # for offline data from data folder
data = prd.prepare_data('GOOGL', distant=True, api_key='your_key') # for realtime data
```

Please check [the available functions](/methods/README.md#methods-package) in the predict file.
All the forecast functions work the same manner and always give a list of values as a prediction.

### Example

```python
import methods.predict as prd
import matplotlib.pyplot as plt

data = prd.prepare_data('GOOGL', distant=True, api_key='XXX') # replace with your key

# Plot data
plt.plot(data['Close'])
plt.show()

# Diffenrentiate data to make it stationnary
diff_data = prd.differentiate(data, 'return-price')

## Use KNN to get predictions
predictions = prd.knn(diff_data, 987)
prd.plot_predictions(data, predictions, 987, diff_order='return-price', print_rms=True)
# in plot_predictions, diff_order='return-price' is mandatory to plot the predictions correctly
# otherwise you will plot the differentiated predictions
```
