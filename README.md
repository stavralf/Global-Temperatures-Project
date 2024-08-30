# Global Temperatures Project

<br/>

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Preproccessing and Descriptive Statistics](#data-preproccessing-and-descriptive-statistics)
- [Time Series Analysis - Yearly](#time-series-analysis---yearly)
    - [Stationarity, Seasonality and Model Selection](#stationarity-seasonality-and-model-selection)
    - [Testing our model](#testing-our-model)
    - [Model Evaluation](#model-evaluation)
- [Monthly Investigation](#monthly-investigation)
    - [Stationarity, Seasonality and Model Selection](#stationarity-seasonality-and-model-selection)
    - [ACF and PACF Plots](#acf-and-pacf-plots)
    - [Model Evaluation](#model-evaluation)
    - [Predictions](#predictions)
##

<br/><br/>

### Project Overview
Taking advantage of the Global_Temperatures_By_Major_City dataset, we perform basic EDA to locate a city of interest for analysing further its temperature history and 
predicting both yearly and monthly temperature figures. A detailed Time Series analysis including various methods for checking the existence of stationarity and applying series transformations if needed, using custom-made tools of differencing. Moreover, we measure and potentially mitigate aurocorrelation between the series lags by analysing ACF anf PACF plots and introducing AutoRegressive and MovingAverage parameters into a framework of ARIMA modelling, accordingly. The afforementioned process is divided into a yearly and monthly temperature investigation, for the purpose of which both a seasonal and a non-seasonal model is built. Throughout and after designing the models, we constantly analyse the behaviour of the residuals for staying alligned with the Gaussian context in which the model employement is taking place, using various plots and techniques for distribution analysis. Finally, metrics of assessment are developed, where along with standard criteria of evaluating time series forecasting accuracy, we establish a versatile landscape of reasoning for approving or not the relevant outcomes. 

### Data Sources 
The dataset employed for our analysis is the "GlobalLandTemperatures_GlobalLandTemperaturesByMajorCity.csv", freely accessible online.

### Tools
The project is materialized with Python in Jupyter Notebook environment. The libraries used are:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Statistical Data Visualisation
from datetime import datetime # To turn strings into dates and other date/time tranformations
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
```

### Data Preproccessing and Descriptive Statistics
 <br/>
 
We first load the dataset and check the type of the variables included.

```python
temps = pd.read_csv('GlobalLandTemperatures_GlobalLandTemperaturesByMajorCity.csv')
temps.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 239177 entries, 0 to 239176
Data columns (total 7 columns):
No.  Column                         Non-Null Count   Dtype  
---  ------                         --------------   -----  
 0   dt                             239177 non-null  object 
 1   AverageTemperature             228175 non-null  float64
 2   AverageTemperatureUncertainty  228175 non-null  float64
 3   City                           239177 non-null  object 
 4   Country                        239177 non-null  object 
 5   Latitude                       239177 non-null  object 
 6   Longitude                      239177 non-null  object 
dtypes: float64(2), object(5)
memory usage: 12.8+ MB
```
<br/>

Null values of each variable. We will deal with them below.

```python
temps.isnull().sum()
```
|Variable | Null |
|--|--|
|dt|                                   0|
|AverageTemperature|               11002|
|AverageTemperatureUncertainty|    11002|
|City|                                 0|
|Country|                              0|
|Latitude|                             0|
|Longitude|                            0|
|dtype: int64|                          |

<br/>

Descriptive Statistics of the numerical variables for each city. Something that stands out is the significantly varying number
of observations for each city.

```python
temps.groupby('City')['AverageTemperature'].describe()
```
|City	   |count|	mean|	std|	min|	25%|	50%|	75%|	max|
|--------|-----|------|----|-----|-----|-----|-----|-----|								
|Abidjan|	1777.0|	26.163737|	1.403715|	22.363|	25.11400|	26.2240|	27.18300|	29.923|
|Addis Abeba|	1679.0|	17.525073|	1.223339|	14.528|	16.56900|	17.2930|	18.47300|	21.223|
|Ahmadabad|	2448.0|	26.529853|	4.260933|	16.792|	22.92325|	27.2575|	29.57450|	35.419|
|Aleppo|	2479.0|	17.370587|	8.536599|	0.670|	9.26050|	17.6120|	25.71900|	32.629|
|Alexandria|	2666.0|	20.312617|	4.559545|	10.227|	15.98725|	20.4635|	24.61250|	28.806|
|...|	...|	...|	...|	...|	...|	...|	...|	...|
|Tokyo|	2020.0|	12.555998|	8.230291|	-1.580|	4.51450|	13.1115|	20.16900|	27.295|
|Toronto|	3141.0|	5.773911|	10.050773|	-15.502|	-3.95500|	5.9150|	15.52900|	25.649|
|Umm Durman|	1768.0|	29.081291|	3.747367|	18.508|	26.11250|	30.2655|	31.79150|	35.700|
|Wuhan|	2072.0|	16.830944|	8.810197|	-0.305|	8.66900|	17.4015|	25.10025|	31.233|
|Xian|	2126.0|	11.487148|	9.410227|	-6.418|	3.24475|	11.8380|	20.68975|	26.762|
|100 rows × 8 columns|

<br/>

```python
obsv_per_city = temps.groupby('City')['AverageTemperature'].count().sort_values()
obsv_per_city.plot.barh(figsize = (16,20))
plt.axvline(np.mean(obsv_per_city.values), color = 'orange')
plt.title('No.of Observations')
plt.xlabel('#')
plt.show()
```
![plot1](https://github.com/user-attachments/assets/8fa6e531-851c-4b79-867b-fc84a3f9a820)

<br/>

At first, we aim to find a specific city to focus our analysis on. 
A look at the existence of null values in our dataset (Number of null values over the total number of entries per city).
The orange line represents the mean null rate.

```python
null_rate = temps.groupby('City')['AverageTemperature'].apply(lambda x : (x.isnull().sum()/x.count())*100)
plt.figure(figsize = (14,20))
null_rate.sort_values().plot.barh()
plt.axvline(np.mean(null_rate[:].values), color = 'orange')
plt.xlabel('%', size = 12)
plt.title('Rate of null values per city', size = 14)
plt.grid(axis = 'x')
plt.show()

pd.DataFrame({"Montreal (%)" : null_rate[null_rate.index == "Montreal"], "Average (%)" : np.mean(null_rate[:].values)})
```
![plot2](https://github.com/user-attachments/assets/85c568ca-9b82-464f-86ff-131eecd2dda8)

|Montreal (%)|	Average (%)|
|------------|-------------|
|3.120025|	5.291562|

<br/><br/>


```python
def twentieth_century(row):
    if row['year'] >= 1900:
        return 1
    else:
        return 0
temps['twentieth_century'] = temps.apply(twentieth_century, axis = 1)
```

<br/>

We have created a dummy variable indicating whether or not the date is before or after 1900, we then group on the basis of this 
variable and we plot the results. The city that stands out from the rest is Montreal, which has witnessed an increase of almost
one Celsius degree on its average temperature after 1900 compared to the time period before.

Looking back at the null value rate plot, Montreal has significantly less null values concetration in its Average Temperature
measurement feature. Thus, Montreal is a reasonable city selection for procceeding our analysis on, this is additionally justified
by the short null value exploration below. 

```python
temp_diff = temps.groupby(["City","twentieth_century"])["AverageTemperature"].mean().unstack()
temp_diff['diff'] = temp_diff[1] - temp_diff[0]

plt.figure(figsize = (14,20))
temp_diff['diff'].sort_values().plot.barh()
plt.xlabel('Difference (Celcius)')
plt.title('Difference of Average Temperature before and since the 20th century', size = 16)
plt.grid(axis = 'x')
```

![plot3](https://github.com/user-attachments/assets/2767b434-67e6-4300-9a42-cd638a202704)

<br/>

From below, all missing temperature values for Montreal are between the years of 1743 and 1780. However, from previous plots, Montreal's
null value rate is significantly below the average (3.1% compared to 5.3%) and the available observations are more than most of the
remaining cities. So we decide to move on our investigation with Montreal by also ommiting its null values.

```python
montreal_pre = temps[temps["City"] == "Montreal"]
montreal_pre.loc[montreal_pre["AverageTemperature"].isnull() == True,'year'].unique()
```

array([1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1754,
       1755, 1756, 1757, 1760, 1761, 1762, 1763, 1764, 1765, 1767, 1774,
       1778, 1780], dtype=int64)

<br/>

We generate the same barplot, but this time only with the years after 1780. We see that Montreal remains on the top of the
temperature change before and after 1900.

```python
temp_diff = temps[temps['year']>1780].groupby(["City","twentieth_century"])["AverageTemperature"].mean().unstack()
temp_diff['diff'] = temp_diff[1] - temp_diff[0]

plt.figure(figsize = (14,20))
temp_diff['diff'].sort_values().plot.barh()
plt.xlabel('Difference (Celcius)')
plt.title('Difference of Average Temperature before and after 1900 (1780-2013)', size = 16)
plt.grid(axis = 'x')
```
![plot4](https://github.com/user-attachments/assets/7c13bcce-681c-40eb-ad1e-62a3948c3835)

<br/>

Moreover, we see that the year of 2013 is not complete, instead the calculations stop in September. 

```python
montreal_pre.tail()
```

|	dt	|AverageTemperature|	AverageTemperatureUncertainty|	City|	Country|	Latitude|	Longitude|	year|	month|	twentieth_century|
|-----|------------------|-------------------------------|------|--------|----------|----------|------|------|-------------------|
|	2013-05-01|	14.065|	0.247|	Montreal|	Canada|	45.81N|	72.69W|	2013|	5|	1|
|2013-06-01	|16.949	|0.286	|Montreal	|Canada	|45.81N	|72.69W	|2013	|6	|1|
|2013-07-01	|21.298|	0.344	|Montreal	|Canada	|45.81N	|72.69W	|2013	|7	|1|
|2013-08-01	|18.738	|0.390	|Montreal	|Canada	|45.81N	|72.69W	|2013	|8	|1|
|2013-09-01	|14.281	|1.110	|Montreal	|Canada	|45.81N	|72.69W	|2013	|9	|1|


<br/>

For the shake of our following analysis, we tranform features related to time into the appropriate datetime format and we create two more 
variables, the month and the year of the temeprature measurement.

```python
temps['dt']  = pd.to_datetime(temps['dt'])
temps['year'] = pd.to_datetime(temps['dt']).dt.year
temps['month'] = pd.to_datetime(temps['dt']).dt.month
```

<br/>

### Time Series Analysis - Yearly

#### Stationarity, Seasonality and Model Selection

Lets examine when Montreal experienced the greatest increase in its yearly average temperature figure.
We first plot the temperature development throughtout the years under investigation. We then check the temperature uncertainty for
the same years. We observe that there is an increasing trend in the average temperature  per year however there are constant  
fluctuations. Moreover, the significant decrease in the standard deviation of the average temperature(uncertainty) calculations is 
a positive sign for the validity of the further stochastic investigation.

```python
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (24,8), sharex = True)

axs[0].plot(temps[(temps["City"] == "Montreal") & (temps["year"] >1780) ].groupby('year')['AverageTemperature'].mean(), color = 'firebrick')
axs[0].set_ylabel('Average Temperature', size = 12)
axs[0].set_title('Evolution of Average Temperature in Montreal per year', size = 16)

axs[1].set_title('Evolution of Average Temperature Uncertainty in Montreal per year', size = 16)
axs[1].plot(temps[(temps["City"] == "Montreal") & (temps["year"] >1780)].groupby('year')['AverageTemperatureUncertainty'].mean(), color = 'black')
axs[1].set_ylabel('Average Temperature\n Uncertainty', size = 12)
axs[1].set_xlabel('Year', size = 12)

plt.show()
```
![plot5](https://github.com/user-attachments/assets/4924342e-f464-4a47-815c-509990d535e7)

<br/>

The Stationarity of the time-series seems not to be established, mainly due to the presence of trend. Another way to obtain
a clear view on the matter is by checking the evolution of mean and variance along time so that any trend in these plots will
indicate clear signs of non-statinarity of the series. Stationarity condition is in the core of time-series analysis, as in part
guarantees greater forecasting ability through standard ARIMA modelling.

```python
montreal = temps[(temps["City"] == "Montreal")& (temps["year"] >1780)].groupby('year')['AverageTemperature'].mean()# Remove na values 

def cummulative_mean(df):
    col = [0 for i in range(len(df))]
    col[0] = df.iloc[0]
    for i in range(1,len(df),1) :
        col[i] = df.iloc[i] + col[i-1]
    for j in range(1,len(df),1):
        col[j] = col[j]/(j+1)
    return col

def cummulative_variance(df):
    col = cummulative_mean(df)
    col_var = [0 for i in range(len(df))]
    for i in range(1,len(df),1) :
        for j in range(0,i,1):
            col_var[i] += ((df.iloc[j] - col[i])**2)/i
    return col_var

# Not used at the moment.
def autocorrelation(df, lag):
    tot_var = np.var(df.iloc[:])*(len(df)-1)
    tot_mean = np.mean(df.iloc[:])
    auto_corr = [1 for i in range(0,lag,1)]
    for j in range(1,lag-1,1):
        for i in range(j+1,len(df)):
            auto_corr[j+1] += (df.iloc[i]-tot_mean)*(df.iloc[i-j]-tot_mean)
    auto_corr[1:] = [auto_corr[i]/tot_var for i in range(1,lag,1)]
    return auto_corr
```

Preparing the dataframe needed for plotting the cummulative mean and cummulative variance of our time series in the following step.

```python
montreal_df = montreal.reset_index()# Turning 'year' into a feature rather than an index. 
montreal_df["mean_stationarity"] = cummulative_mean(montreal)
montreal_df['var_stationarity'] = cummulative_variance(montreal)
```

It is witnessed from the two linegraphs below, that there is a dependency on time (or a trend) in both the mean and 
the variance of temperature values. Especially, in both lines we can see a strong increasing trend in the years after 1900.
Therefore, differencing the series for establishing stationarity might be needed, so we embark on this examination in the 
following steps.

```python
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (24,12), sharex = False)
fig.suptitle("Stationarity Check", size = 16)

x_1 = montreal_df['year'];y_1 = montreal_df['mean_stationarity'];y_2 = montreal_df['var_stationarity']

axs[0].plot(x_1,y_1, color = 'firebrick')
axs[1].plot(x_1, y_2, color = 'black')
axs[0].set_ylabel('Temperature (Celcius)', size = 12)
axs[1].set_ylabel('Temperature (Celcius)', size = 12)
axs[1].set_xlabel('Year',size = 12)
axs[0].set_title("Cummulative Mean of Montreal's Temperature", size = 14)
axs[1].set_title("Cummulative Variance of Montreal's Temperature", size = 14)
axs[0].grid(axis = 'x')
axs[1].grid(axis = 'x')
```

![download](https://github.com/user-attachments/assets/ed8f9c8b-8bcb-4a4b-898c-d82a6da1b39d)

<br/>

We employ the tactic of subtracting the time series with its own past values, the latter being described through the variable
'times' or how many subtractions do we perform. Below we build the differencing functions.

```python
def differencing(df):
    differenced = [0 for i in range(0,len(df)-1,1)]
    for i in range(len(df)-1):
            differenced[i] = df.iloc[i+1] - df.iloc[i]
    return pd.Series(index = df.index[1:] ,data = differenced, name = 'AverageTemperature')

def multiple_differencing(df, times):
    for i in range(times):
        df = differencing(df)
    return df
```

We now apply differencing and we then check the time-series plots of the resulted series from which we see that 
the presence of the trend is mitigated and thus stationarity is a possibility that we investigate further.

```python
colors = ['black', 'firebrick', 'green','orange']
fig,axs = plt.subplots(ncols = 1,nrows = 4, figsize =(24,20), sharex = 'row')
axs[3].set_xlabel('Year')

for times in range(4):
    montreal_differenced = multiple_differencing(montreal,times)
    axs[times].plot(montreal_differenced, color = colors[times])
    axs[times].axhline(np.mean(montreal_differenced.values))
    axs[times].set_ylabel('Celcius')
    axs[times].set_title(f'd = {times}', size =12)

fig.suptitle('d-Times Differenced Time Series', size = 18)
plt.show()
```
![download](https://github.com/user-attachments/assets/51715032-1213-4334-b29b-b5ec3672ee3a)

<br/>

After differencing the series once, we continue by investigating the Auto-Correlation and the Patial Auto-Correlation between 
the present value and the past values(lags) of the stochastic process by way of checking the sufficiency of the differencing order
and start determining possible values for the parameters suitable for the ARIMA model. Regarding the order of difference as the 
ACF plot is patternless, the 1-lag has correlation magnitude less than 0.5 and (almost) all the rest of the lags are insignificant, 
a first order differencing seems sufficient. On the other hand, the PACF plot demonstrates correlation with 2- and 3-lag as well, 
which is not justified by the ACF plot. Hence we decide to focus on the piece of information given by the ACF plot and see if 
the Partial correlation has been mitigated.

Moreover, from the ACF plot, the negative correlation in 1-lag and the following sharp cut-off in the following lags indicates
that the series might be slightly over-differenced so that we may proceed by adding a Moving Average (MA) term to the model.

<br/>
Differencing the series.

```python
montreal_differenced = multiple_differencing(montreal,1)
```

Plot the ACF & PACF plots

```python
fig,axs = plt.subplots(nrows = 1,ncols=2, figsize = (16,4))
plot_acf(montreal_differenced, ax = axs[0])
plot_pacf(montreal_differenced, ax = axs[1])
plt.show()
```
![download](https://github.com/user-attachments/assets/78c59c78-a782-4b95-b6c3-df61e70f9812)

<br/>

The ACF and PACF plots below are showing that the correlation present in the previous plots has been eliminated after the addition of the
MA term. Even though the residuals plot is showing a slightly worrisome wandering away from the mean (see next block), the variance of the 
residuals has been decreased and the MA term of the model is significantly different to zero. Hence, we decide to continue with this 
model (the model structure is : Y^_t = Y_(t-1) - ma.L1*e_(t-1), where e_(t-1) = Y^_(t-1) - Y_(t-1)).
<br/>
Model

```python
model = ARIMA(endog = montreal.values, order = (0,1,1))
model_fit = model.fit()
pred = model_fit.predict()# In-sample predictions for the whole dataset. To each step the real measurement is added and a prediction is performed. 
res = montreal.values - pred
```
Plot the ACF & PACF plots

```python
fig,axs = plt.subplots(nrows = 1,ncols=2, figsize = (16,4))
plot_acf(res[1:], ax = axs[0])
plot_pacf(res[1:], ax = axs[1])
plt.show()

print(f'Residual Variance : {np.var(res)}')
model_fit.summary()
```
![download](https://github.com/user-attachments/assets/cd5725ba-a444-4818-8a89-8d7922c55b54)

![Screenshot 2024-08-08 140356](https://github.com/user-attachments/assets/18378803-d1f4-4653-80b9-33d691c67c52)

<br/>

#### Testing our model.

Train and Test Set.

```python
montreal_train = montreal[:round(len(montreal)*0.9)].reset_index()# We select 90% of the data available for training
montreal_test = montreal[round(len(montreal)*0.9):].reset_index()
```

In-Sample Forecasting step-by-step.
```python
pred = [0 for i in range(len(montreal_test))]
for i in range(len(montreal_test)):
    model = ARIMA(montreal_train['AverageTemperature'], order = (0,1,1))
    fitted = model.fit()
    pred[i] = fitted.forecast().values[0]
    montreal_train.loc[len(montreal_train)+i,'AverageTemperature'] = montreal_test.loc[i,'AverageTemperature']
    montreal_train.loc[len(montreal_train)+i,'year'] = montreal_test.loc[i,'year']
```

Plot the results. Our model can predict the general increasing trend of the average temperature in the years under consideration,
while it seems that it does not overfit the data given for training. However, it cannot capture the magnitude of the fluctuations
and slightly misses out the year-to-year increase or decrease in the figure of temperature. We however keep in mind that this does 
not constitute a long-term forecasting.

```python
forecast_on_test = pd.Series(index = montreal_test['year'], data = pred)

plt.figure(figsize = (24,8)) 
plt.plot(montreal.iloc[:round(len(montreal)*0.9)], color = 'black')
plt.plot(montreal.iloc[round(len(montreal)*0.9):], color = 'firebrick',label = 'Actual')
plt.plot(forecast_on_test, color = 'orange', label = 'forecasted')
plt.legend()
plt.show()
```
![download](https://github.com/user-attachments/assets/a9c9607c-b3fb-4bdf-b3a7-6cd33571e4b7)

<br/>

#### Model Evaluation

Lets formulate few basic metrics for assessing the performance of our model

```python
def metrics_of_accuracy(pred, actual):
    me = np.mean(pred - actual) # Mean Error
    mae = np.mean(np.abs(pred - actual)) # Mean Absolute Error
    mpe = np.mean((pred - actual)/actual)*100# Mean Percentage Error
    mape = np.mean(np.abs((pred - actual)/actual))*100# Mean Absolute Percentage Error
    corr = np.corrcoef(pred, actual)[0,1]# Correlation between predictions and actual values
    return (pd.DataFrame({'Mean Error' : me, 'Mean Absolute Error' : mae, 'Mean Percentage Error  (%)' : mpe,
             'Mean Absolute Percentage Error (%)' :  mape, 'Correlation' : corr}, index = [0]))

metrics_of_accuracy(pred,montreal.values[round(len(montreal)*0.9):])
```

![Screenshot 2024-08-08 151135](https://github.com/user-attachments/assets/fedc3c62-be44-443d-b8f5-7566aa2bfcab)

<br/>

We calculate the same accuracy measures but for a range of AR terms to check whether the possibility of adding such terms would
ameliorate the performance. We see that even though there is a small imporvement in the correaltion and the MAPE when an AR term 
is introduced, that is not the case for the ME and MPE. We can investigate introducing AR terms to the model further, but for now,
we proceed with the monthly investigation of the Montreal's temperature.

```python
frames = [0 for i in range(4)]
for i in range(0,4,1):
    montreal_train = montreal[:round(len(montreal)*0.9)].reset_index()# We select 90% of the data available for training
    montreal_test = montreal[round(len(montreal)*0.9):].reset_index()
    pred = [0 for k in range(len(montreal_test))]
    for j in range(len(montreal_test)):
        model = ARIMA(montreal_train['AverageTemperature'], order = (i,1,1))
        fitted = model.fit()
        pred[j] = fitted.forecast().values[0]
        montreal_train.loc[len(montreal_train)+j,'AverageTemperature'] = montreal_test.loc[j,'AverageTemperature']
        montreal_train.loc[len(montreal_train)+j,'year'] = montreal_test.loc[j,'year']
    frames[i] = metrics_of_accuracy(pred,montreal.values[round(len(montreal)*0.9):])

pd.concat(frames).reset_index().drop('index', axis = 1)
```

![Screenshot 2024-08-08 151502](https://github.com/user-attachments/assets/c4bb9483-04b3-41c5-a522-d16585e5d5c8)


<br/>

### Monthly Investigation.

#### Stationarity, Seasonality and Model Selection
Here we start our deroute for investigating Montreal's temperature per month, aiming to prove that the 
standard seasonal/weather correlation between the same months or seasons holds in our dataset as well and to perform 
prediction analysis. Plotting the series below, a clear seasonal pattern is observed while, in the first plot we observe
slight upward trend, in fact slightly less spikes occur in the low temperatures.

```python
montreal_monthly = temps.loc[(temps["City"] == 'Montreal') & (temps['year']> 1780), ['dt','year','month','AverageTemperature']].reset_index().drop('index', axis = 1)

montreal_monthly_series = pd.Series(index = np.array(montreal_monthly['dt']), 
                                    data = np.array(montreal_monthly["AverageTemperature"]),
                                   name = "AverageTemperature")

fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (24,12))

axs[0].plot(montreal_monthly_series, color = 'black')
axs[0].axhline(np.mean(montreal_monthly_series.values), color = 'orange')
axs[1].plot(montreal_monthly_series[montreal_monthly_series.index >= '1950-01-01'],color = 'black')
axs[1].axhline(np.mean(montreal_monthly_series[montreal_monthly_series.index >= '1950-01-01'].values), color = 'orange')
axs[0].set_title('Monthly Temperature in Montreal (1781 - 2013)')
axs[0].set_ylabel('Temperature (Celcius)')
axs[1].set_title('Monthly Temperature in Montreal (1950 - 2013)')
axs[1].set_ylabel('Temperature (Celcius)')
axs[1].set_xlabel('Year')

plt.show()
```
![download](https://github.com/user-attachments/assets/aabd3b90-55d5-40b8-982a-e97122e8e064)

<br/>

Seasonal Differencing. Instead of subtracting a time-series with its 1-Lag, we generalise the concept and we subtract any k-lag
This will allow us to mitigate the seasonal pattern seen in the plot above.

```python
def seasonal_differencing(df, lag) :
    s_differenced = [0 for i in range(0,len(df)-lag,1)]
    for i in range(0,len(df)-lag,1):
            s_differenced[i] = df.iloc[i+lag] - df.iloc[i]
    return pd.Series(index = df.index[lag:] ,data = s_differenced, name = 'AverageTemperature')

def multiple_s_differencing(df,lag, times):
    for i in range(times):
        df = seasonal_differencing(df,lag)
    return df
```

Lets denote the times of non-seasonal differencing by d, and the times of seasonal differencing by D. As there is clear seasonal
pattern and our data consists of monthly temperatures, we start by applying seasonal differencing of period 12 (handled by the
variable 'times' in the functions above). 

```python
montreal_monthly_series_diff = multiple_differencing(montreal_monthly_series,1)# Single Differencing

montreal_monthly_series_s_diff = multiple_s_differencing(montreal_monthly_series,12,1)# Subtracting Average Temperatures every 12 months (D = 1) 

montreal_monthly_series_s_diff_1 = multiple_differencing(montreal_monthly_series_s_diff,1)# Single differencing (d = 1)
```

>Plot the three time series. The original, the one-time seasonally differenced, and finally, the latter non-seasonally differenced once. 
>We can see that the spikes are less present

```python
fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (26,14))

axs[0].plot(montreal_monthly_series, color = 'black')
axs[0].set_title('Original Series')
axs[0].axhline(np.mean(montreal_monthly_series), color = 'orange')
axs[0].set_ylabel('Temperature (Celcius)')

axs[1].plot(montreal_monthly_series_s_diff,color = 'firebrick')
axs[1].axhline(np.mean(montreal_monthly_series_s_diff), color = 'orange')
axs[1].set_title('Seasonal Differencing | lag = 12')
axs[1].set_ylabel('Temperature (Celcius)')

axs[2].plot(montreal_monthly_series_s_diff_1, color = 'green')
axs[2].axhline(np.mean(montreal_monthly_series_s_diff_1), color = 'orange')
axs[2].set_title('Single Differencing and Seasonal Differencing | lag = 12')
axs[2].set_ylabel('Temperature (Celcius)')
axs[2].set_xlabel('Year')
plt.show()
```
![download](https://github.com/user-attachments/assets/cdab9c56-180e-4e5c-86e1-04d9a0df20f3)

#### ACF and PACF Plots
>ACF and PACF plots after single seasonal differencing with 12-lag. The negative lags multiple to 12-lag in the PACF plot imply that
>the series is probalbly slightly 'seasonally over-differenced' and then a SMA term could possibly mitigate this behaviour. Moreover, the 
>positive 1-lag in the PACF implies that the series is 'non-seasonally underdifferenced', hence an AR term could help in this issue. 

```python
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,4))
plot_acf(montreal_monthly_series_s_diff, ax = axs[0])
plot_pacf(montreal_monthly_series_s_diff, ax = axs[1])
plt.show()
```
![download](https://github.com/user-attachments/assets/b29ed1d6-22e3-4c8d-86b3-97282e6444ff)

<br/>

We first check the autocorrelations after only adding an SMA term. We can see that the multipe to 12-lags correaltions have been
eliminated, however there is now an even stronger presence of positive autocorrealtion and partial auto-correlation with the first 
couple of lags. Hence, we employ the afforementioned strategy, namely of adding an AR term.
```python
s_model = ARIMA(endog = montreal_monthly_series, order = (0,0,0), seasonal_order = (0,1,1,12))
s_fitted = s_model.fit()
s_pred = s_fitted.predict()
s_res = montreal_monthly_series.values - s_pred

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,4))
plot_acf(s_res, ax = axs[0])
plot_pacf(s_res, ax = axs[1])
plt.show()

s_fitted.summary()
```

![download](https://github.com/user-attachments/assets/4558b729-664a-435b-b991-528056452598)
![Screenshot 2024-08-08 152514](https://github.com/user-attachments/assets/6cf9fb4e-aebb-4ba5-8860-fb985eca522f)

<br/>

#### Model Evaluation
As the predictions are on float numbers of many decimal places, the MAPE metric creates, almost necessarily, a rather high figure
hence the metric of MAE is a more reliable one to count our assessment on.

```python
s_model = ARIMA(endog = montreal_monthly_series, order = (1,0,0), seasonal_order = (0,1,1,12))
s_fitted = s_model.fit()
s_pred = s_fitted.predict()

plt.figure(figsize = (28,8))
plt.plot(pd.Series(index = montreal_monthly_series.index[2500:], data = s_pred[2500:]))
plt.scatter(y = montreal_monthly_series[2500:], x = montreal_monthly_series.index[2500:], color = 'black')

metrics_of_accuracy(s_pred, montreal_monthly_series.values)
```
![download](https://github.com/user-attachments/assets/2b72cad7-c100-4291-9376-e7e53fa291e4)
![Screenshot 2024-08-08 152706](https://github.com/user-attachments/assets/f094cc06-8ebb-4b6a-94ed-e7d475c00f54)

<br/>

```python
train_monthly_set =  montreal_monthly_series[:len(montreal_monthly_series) - 12].reset_index()
test_monthly_set = montreal_monthly_series[len(train_monthly_set):].reset_index()

model = ARIMA(train_monthly_set["AverageTemperature"], order = (1,0,0), seasonal_order = (0,1,1,12))
fitted = model.fit()
pred = fitted.forecast(steps = 12)
test_monthly_set.index = montreal_monthly_series.index[len(montreal_monthly_series)-12:]
pred.index = montreal_monthly_series.index[len(montreal_monthly_series)-12:]
```
#### Predictions
In the plot below, we gather the out-of-sample forecasts for the last 12 months of our dataset based on a SARIMA model of the form 
y^{-}_t = AR_1*y_(t-1) + w_t + SMA_1*w_(t-12). The result is satisfying, however there is a possibility of overfitting and futher
examination of ways to enhance the model performance might be required. 

```python
plt.figure(figsize = (28,8))
plt.plot(pd.Series(index = montreal_monthly_series.index[2700:], data = montreal_monthly_series.values[2700:]),color = 'blue')
plt.scatter(x = montreal_monthly_series.index[2700:], y = montreal_monthly_series.values[2700:],color = 'blue')
plt.plot(test_monthly_set, color = 'black', label = 'actual')
plt.scatter(x = test_monthly_set.index, y = test_monthly_set.values, color = 'black')
plt.plot(pred, color = 'red', label = 'predicted')
plt.scatter(x = pred.index, y = pred.values, color = 'red')
plt.legend()
plt.show()
```

![download](https://github.com/user-attachments/assets/dca50ff3-1047-441d-9927-4d62531b6b4a)

##
<br/><br/>

The whole script of this project can be found [here.](https://github.com/stavralf/Global-Temperatures-Project/blob/main/Global_Temperatures_Analysis.ipynb)  



