# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 5
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Loading the datasets
train = pd.read_csv('dataset/yds_train2018.csv')
test = pd.read_csv('dataset/yds_test2018.csv')
promo_expense = pd.read_csv('dataset/promotional_expense.csv')
holidays = pd.read_excel('dataset/holidays.xlsx')

# Defining the evaluationg function
def sMAPE(forcasted, actual):
    return abs(forcasted-actual)/(abs(forcasted)+abs(actual))

# Function to combine month and year
def monthsConvert(x):
    if int(x) in range(1,10):
        return "0"+x
    else:
        return x
    
# Evaluating Rolling Statistics and Dickey-fuller test
def test_stationary(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window = 12).mean()
    rolstd = timeseries.rolling(window = 12).std()
    #print(rolmean,rolstd)
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Dictionery to store unique country and product id data
train_dict = {}

# Copying the test file
test_data = test.copy()


# Getting unique months
data_monthly = train.groupby(['S_No','Year', 'Month', 'Product_ID', 'Country']).agg({'Sales':sum})
data_monthly.reset_index(level=[ 'Year', 'Month', 'Product_ID', 'Country'], inplace =True)
data_monthly.head()

# Getting unique country
uniqCountry = data_monthly['Country'].unique()

for country in uniqCountry:
    data_month_country = data_monthly[data_monthly['Country']==country]
    uniqProd = data_month_country['Product_ID'].unique()
    
    # Getting unique product id 
    for product in uniqProd:
        
        # Training data group by Sales with index in timestamp format 
        data_month_filtered = data_month_country[data_month_country['Product_ID']==product]
        data_month_filtered['time'] = data_month_filtered['Year'].astype(str) +"-" + data_month_filtered['Month'].astype(str).apply(monthsConvert)
        data_month_filtered['time'].apply(lambda dates: pd.datetime.strptime(dates, '%Y-%m'))
        datagroupby = data_month_filtered.groupby(['time']).agg({'Sales':sum})
        datagroupby.index= pd.to_datetime(datagroupby.index)
        train_dict[(country,product)] = datagroupby['Sales']    
        
        #Test data group by Sales with index in timestamp format 
        test_data = test[(test['Country']== country) & (test['Product_ID']==product)]
        test_data['time'] = test_data['Year'].astype(str) +"-" + test_data['Month'].astype(str).apply(monthsConvert)
        test_data['time'].apply(lambda dates: pd.datetime.strptime(dates, '%Y-%m'))
        test_data = test_data.groupby(['time']).agg({'Sales':sum})
        test_data.index= pd.to_datetime(test_data.index)
        
## Adding the time column in test data
#test_data['time'] = test_data['Year'].astype(str) +"-" + test_data['Month'].astype(str).apply(monthsConvert)
#test_data['time'].apply(lambda dates: pd.datetime.strptime(dates, '%Y-%m'))


for key,value in train_dict:
    print((key,value),end=" ")

##-----------------------------('Argentina', 1)--------------------------------##

val_1 = np.power(train_dict[('Finland', 4) ],1)
train_1, test_1 = train_test_split(val_1, test_size=0.15,shuffle=False)

test_data_1 = test[(test['Country']== 'Finland') & (test['Product_ID']==4)]

model_1 = ExponentialSmoothing(train_1, seasonal='add', trend = 'add', seasonal_periods=11).fit()
pred_1 = model_1.predict(start=test_1.index[0], end=test_1.index[-1])
plt.plot(train_1.index, train_1, label='Train')
plt.plot(test_1.index, test_1, label='Test')
plt.plot(pred_1.index, pred_1, label='Holt-Winters')
plt.legend(loc='best')
plt.show()
error = sMAPE(pred_1,test_1).mean()
print(error)

test_pred_1 = pred_1.predict(start=test)

##-----------------------------('Finland', 4)--------------------------------##

from scipy.special import boxcox

val_1 = np.power(train_dict[('Finland', 4) ],1/4)
train_1, test_1 = train_test_split(val_1, test_size=0.15,shuffle=False)

test_data_1 = test[(test['Country']== 'Finland') & (test['Product_ID']==4)]

for i in ['add','mul']:
    for j in ['add','mul']:
        for k in range(2,13):
            
            model_1 = ExponentialSmoothing(train_1, seasonal=i, trend = j, seasonal_periods=k).fit()
            pred_1 = model_1.predict(start=test_1.index[0], end=test_1.index[-1])
            plt.plot(train_1.index, train_1, label='Train')
            plt.plot(test_1.index, test_1, label='Test')
            plt.plot(pred_1.index, pred_1, label='Holt-Winters')
            plt.legend(loc='best')
            plt.show()
            error = sMAPE(pred_1,test_1).mean()
            print(error)