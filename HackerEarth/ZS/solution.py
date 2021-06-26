import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 5

train = pd.read_csv('dataset/yds_train2018.csv')
test = pd.read_csv('dataset/yds_test2018.csv')
promo_expense = pd.read_csv('dataset/promotional_expense.csv')
holidays = pd.read_excel('dataset/holidays.xlsx')

def sMAPE(forcasted, actual):
    return abs(forcasted-actual)/(abs(forcasted)+abs(actual))

data_monthly = train.groupby(['S_No','Year', 'Month', 'Product_ID', 'Country']).agg({'Sales':sum})
data_monthly.reset_index(level=[ 'Year', 'Month', 'Product_ID', 'Country'], inplace =True)
data_monthly.head()

uniqCountry = data_monthly['Country'].unique()
def monthsConvert(x):
    if int(x) in range(1,10):
        return "0"+x
    else:
        return x

ts = {}

from statsmodels.tsa.stattools import adfuller
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
    
for country in uniqCountry:
    data_month_country = data_monthly[data_monthly['Country']==country]
    uniqProd = data_month_country['Product_ID'].unique()
    for product in uniqProd:
        data_month_filtered = data_month_country[data_month_country['Product_ID']==product]
        data_month_filtered['time'] = data_month_filtered['Year'].astype(str) +"-" + data_month_filtered['Month'].astype(str).apply(monthsConvert)
        data_month_filtered['time'].apply(lambda dates: pd.datetime.strptime(dates, '%Y-%m'))
        datagroupby = data_month_filtered.groupby(['time']).agg({'Sales':sum})
        datagroupby.index= pd.to_datetime(datagroupby.index)
        #print(datagroupby.index)
        ts[(country,product)] = datagroupby['Sales']    

for key,value in ts:
    print((key,value),end=" ")

from sklearn.model_selection import train_test_split

train, test = train_test_split(ts[('Argentina', 1)], test_size=0.2,shuffle=False)

test[(test['Country']== 'Argentina') & (test['Product_ID']==2)]

##-----------------------------('Argentina', 1)--------------------------------##

from scipy.special import boxcox
ts[('Argentina', 1)]

ts_log = np.power(ts[('Argentina', 1)],3)
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
test_stationary(ts_log_diff)

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF: 
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
#plt.axvline(x=0.5,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
#plt.axvline(x=0.5,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(1,0,0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
result = sMAPE(results_AR.fittedvalues,ts_log_diff).mean()
plt.title('RSS: %.4f'% result)

model = ARIMA(ts_log, order=(0, 0, 1))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
result = sMAPE(results_MA.fittedvalues,ts_log_diff).mean()
plt.title('RSS: %.4f'% result)

model = ARIMA(ts_log, order=(1,0,1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
result = sMAPE(results_ARIMA.fittedvalues,ts_log_diff).mean()
plt.title('RSS: %.4f'% result)

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA = np.power(predictions_ARIMA_diff,1/3)
plt.plot(ts[('Argentina', 1)])
plt.plot(predictions_ARIMA)
#print(predictions_ARIMA)
result = sMAPE(predictions_ARIMA,ts[('Argentina', 1)]).mean()
plt.title('RSS: %.4f'% result)

results_ARIMA.plot_predict(1,51)

x= results_ARIMA.forecast(steps=8)
print(np.power(x[0],1/3))

##-----------------------------('Argentina', 2)--------------------------------##
print(ts[('Argentina', 2)])
ts_log = np.sqrt(ts[('Argentina', 2)])

ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
test_stationary(ts_log_diff)

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

model = ARIMA(ts_log, order=(1,1,0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
result = sMAPE(results_AR.fittedvalues,ts_log_diff).mean()
plt.title('RSS: %.4f'% result)

model = ARIMA(ts_log, order=(0, 1, 1))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
result = sMAPE(results_MA.fittedvalues,ts_log_diff).mean()
plt.title('RSS: %.4f'% result)

model = ARIMA(ts_log, order=(1,1,1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
result =sMAPE(results_ARIMA.fittedvalues,ts_log_diff).mean()
plt.title('RSS: %.4f'% result)


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.power(predictions_ARIMA_log,2)
plt.plot(ts[('Argentina', 2)])
plt.plot(predictions_ARIMA)
#print(predictions_ARIMA)
result = sMAPE(predictions_ARIMA,ts[('Argentina', 2)]).mean()
plt.title('RSS: %.4f'% result)

results_ARIMA.plot_predict(1,51)
del x
x= results_ARIMA.forecast(steps=12)
print(np.power(x[0],2))

##-----------------------------('Argentina', 3)--------------------------------##

ts_log3 = np.sqrt(ts[('Argentina', 3)])

ts_log_diff3 = ts_log3 - ts_log3.shift()
ts_log_diff3.dropna(inplace=True)
test_stationary(ts_log_diff3)


lag_acf3 = acf(ts_log_diff3, nlags=20)
lag_pacf3 = pacf(ts_log_diff3, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf3)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff3)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff3)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf3)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff3)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff3)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


model = ARIMA(ts_log3, order=(2,1,0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff3)
plt.plot(results_AR.fittedvalues, color='red')
result = sMAPE(results_AR.fittedvalues,ts_log_diff3).mean()
plt.title('RSS: %.4f'% result)

model = ARIMA(ts_log3, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff3)
plt.plot(results_MA.fittedvalues, color='red')
result = sMAPE(results_MA.fittedvalues,ts_log_diff3).mean()
plt.title('RSS: %.4f'% result)

model = ARIMA(ts_log3, order=(1,2,1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff3)
plt.plot(results_ARIMA.fittedvalues, color='red')
result =sMAPE(results_ARIMA.fittedvalues,ts_log_diff3).mean()
plt.title('RSS: %.4f'% result)


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(ts_log3.ix[0], index=ts_log3.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.power(predictions_ARIMA_log,2)
plt.plot(ts[('Argentina', 3)])
plt.plot(predictions_ARIMA)
#print(predictions_ARIMA)
result = sMAPE(predictions_ARIMA,ts[('Argentina', 2)]).mean()
plt.title('RSS: %.4f'% result)

results_ARIMA.plot_predict(1,51)
del x
x= results_ARIMA.forecast(steps=12)
print(np.power(x[0],2))