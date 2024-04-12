# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DSLab Homework 1 - Data Science with CO2
#
# ## Hand-in Instructions
#
# - __Due: 19.03.2024 23h59 CET__
# - `./setup.sh` before you can start working on this notebook.
# - `git push` your final verion to the master branch of your group's Renku repository before the due date.
# - check if `environment.yml` and `requirements.txt` are properly written
# - add necessary comments and discussion to make your codes readable

# %% [markdown]
# ## Carbosense
#
# The project Carbosense establishes a uniquely dense CO2 sensor network across Switzerland to provide near-real time information on man-made emissions and CO2 uptake by the biosphere. The main goal of the project is to improve the understanding of the small-scale CO2 fluxes in Switzerland and concurrently to contribute to a better top-down quantification of the Swiss CO2 emissions. The Carbosense network has a spatial focus on the City of Zurich where more than 50 sensors are deployed. Network operations started in July 2017.
#
# <img src="http://carbosense.wdfiles.com/local--files/main:project/CarboSense_MAP_20191113_LowRes.jpg" width="500">
#
# <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_ZLMT_3.JPG" width="156">  <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_sensor_SMALL.jpg" width="300">

# %% [markdown]
# ## Description of the homework
#
# In this homework, we will curate a set of **CO2 measurements**, measured from cheap but inaccurate sensors, that have been deployed in the city of Zurich from the Carbosense project. The goal of the exercise is twofold: 
#
# 1. Learn how to deal with real world sensor timeseries data, and organize them efficiently using python dataframes.
#
# 2. Apply data science tools to model the measurements, and use the learned model to process them (e.g., detect drifts in the sensor measurements). 
#
# The sensor network consists of 46 sites, located in different parts of the city. Each site contains three different sensors measuring (a) **CO2 concentration**, (b) **temperature**, and (c) **humidity**. Beside these measurements, we have the following additional information that can be used to process the measurements: 
#
# 1. The **altitude** at which the CO2 sensor is located, and the GPS coordinates (latitude, longitude).
#
# 2. A clustering of the city of Zurich in 17 different city **zones** and the zone in which the sensor belongs to. Some characteristic zones are industrial area, residential area, forest, glacier, lake, etc.
#
# ## Prior knowledge
#
# The average value of the CO2 in a city is approximately 400 ppm. However, the exact measurement in each site depends on parameters such as the temperature, the humidity, the altitude, and the level of traffic around the site. For example, sensors positioned in high altitude (mountains, forests), are expected to have a much lower and uniform level of CO2 than sensors that are positioned in a business area with much higher traffic activity. Moreover, we know that there is a strong dependence of the CO2 measurements, on temperature and humidity.
#
# Given this knowledge, you are asked to define an algorithm that curates the data, by detecting and removing potential drifts. **The algorithm should be based on the fact that sensors in similar conditions are expected to have similar measurements.** 
#
# ## To start with
#
# The following csv files in the `../data/carbosense-raw/` folder will be needed: 
#
# 1. `CO2_sensor_measurements.csv`
#     
#    __Description__: It contains the CO2 measurements `CO2`, the name of the site `LocationName`, a unique sensor identifier `SensorUnit_ID`, and the time instance in which the measurement was taken `timestamp`.
#     
# 2. `temperature_humidity.csv`
#
#    __Description__: It contains the temperature and the humidity measurements for each sensor identifier, at each timestamp `Timestamp`. For each `SensorUnit_ID`, the temperature and the humidity can be found in the corresponding columns of the dataframe `{SensorUnit_ID}.temperature`, `{SensorUnit_ID}.humidity`.
#     
# 3. `sensor_metadata_updated.csv`
#
#    __Description__: It contains the name of the site `LocationName`, the zone index `zone`, the altitude in meters `altitude`, the longitude `LON`, and the latitude `LAT`. 
#
# Import the following python packages:

# %%
import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import plotly.graph_objects as go
import os

# %%
pd.options.mode.chained_assignment = None

# %% [markdown]
# ## PART I: Handling time series with pandas (10 points)

# %%
DATA_DIR = '../data/'

# %%
co2_df = pd.read_csv(DATA_DIR + "CO2_sensor_measurements.csv", sep='\t')
print(co2_df.shape)
co2_df.head(9)

# %%
temp_df = pd.read_csv(DATA_DIR + "temperature_humidity.csv", sep='\t')
print(temp_df.shape)
temp_df.head(3)

# %%
sensors_df = pd.read_csv(DATA_DIR + "sensors_metadata_updated.csv", sep=',', index_col=[0])
print(sensors_df.shape)
sensors_df.head(3)

# %% [markdown]
# ### a) **8/10**
#
# Merge the `CO2_sensor_measurements.csv`, `temperature_humidity.csv`, and `sensors_metadata.csv`, into a single dataframe. 
#
# * The merged dataframe contains:
#     - index: the time instance `timestamp` of the measurements
#     - columns: the location of the site `LocationName`, the sensor ID `SensorUnit_ID`, the CO2 measurement `CO2`, the `temperature`, the `humidity`, the `zone`, the `altitude`, the longitude `lon` and the latitude `lat`.
#
# | timestamp | LocationName | SensorUnit_ID | CO2 | temperature | humidity | zone | altitude | lon | lat |
# |:---------:|:------------:|:-------------:|:---:|:-----------:|:--------:|:----:|:--------:|:---:|:---:|
# |    ...    |      ...     |      ...      | ... |     ...     |    ...   |  ... |    ...   | ... | ... |
#
#
#
# * For each measurement (CO2, humidity, temperature), __take the average over an interval of 30 min__. 
#
# * If there are missing measurements, __interpolate them linearly__ from measurements that are close by in time.
#
# __Hints__: The following methods could be useful
#
# 1. ```python 
# pandas.DataFrame.resample()
# ``` 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
#     
# 2. ```python
# pandas.DataFrame.interpolate()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
#     
# 3. ```python
# pandas.DataFrame.mean()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html
#     
# 4. ```python
# pandas.DataFrame.append()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html

# %%
co2_df['timestamp'] = pd.to_datetime(co2_df['timestamp'])
temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'])

# %%
temp_df = temp_df.resample('30T', on='Timestamp').mean().reset_index()
temp_df.head()

# %%
co2_df = co2_df.groupby(['LocationName', 'SensorUnit_ID']).resample('30T', on='timestamp').mean(numeric_only=True)['CO2'].reset_index()

# %%
co2_df.head()

# %%
melted_even_df = pd.melt(temp_df.iloc[:, ::2], id_vars=['Timestamp'], var_name='SensorUnit_ID', value_name='humidity')
melted_odd_df = pd.melt(temp_df.iloc[:, 1::2], value_name='temperature')

temp_df = melted_even_df.join(melted_odd_df)
temp_df.drop(columns=['variable'], inplace=True)
temp_df['SensorUnit_ID'] = temp_df['SensorUnit_ID'].apply(lambda x: int(x.split('.')[0]))

# %%
temp_df.head()

# %%
print(temp_df.shape)

# %%
merged_df = pd.merge(temp_df, co2_df, left_on=['Timestamp', 'SensorUnit_ID'], right_on=['timestamp', 'SensorUnit_ID'], how='outer')

# %%
merged_df.head()

# %%
merged_df = pd.merge(merged_df, sensors_df, on='LocationName', how='inner')

# %%
merged_df.head()

# %%
merged_df = merged_df[['timestamp', 'LocationName', 'SensorUnit_ID', 'CO2', 'temperature', 'humidity', 'zone', 'altitude', 'LON', 'LAT']]
merged_df.set_index('timestamp', inplace=True)
merged_df.head()

# %%
index = merged_df.isnull().any(axis=1)

# %%
merged_df[index]

# %%
merged_df.interpolate(method='linear', inplace=True)

# %%
merged_df[index]

# %%
merged_df.shape

# %% [markdown]
# ### b) **2/10** 
#
# Export the curated and ready to use timeseries to a csv file, and properly push the merged csv to Git LFS.

# %%
merged_df.to_csv(DATA_DIR + 'CO2_measurements_complete.csv', index=False)

# %% [markdown]
# ## PART II: Data visualization (15 points)

# %% [markdown]
# ### a) **5/15** 
# Group the sites based on their altitude, by performing K-means clustering. 
# - Find the optimal number of clusters using the [Elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)). 
# - Wite out the formula of metric you use for Elbow curve.
# - Perform clustering with the optimal number of clusters and add an additional column `altitude_cluster` to the dataframe of the previous question indicating the altitude cluster index. 
# - Report your findings.
#
# __Note__: [Yellowbrick](http://www.scikit-yb.org/) is a very nice Machine Learning Visualization extension to scikit-learn, which might be useful to you. 

# %% [markdown]
# __Answer:__ $ Distortion = ...

# %%
merged_df['altitude'].unique()

# %%

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %%
altitude_data = merged_df[['altitude']]

# %%
altitude_data

# %%
sse = {}  
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42,n_init='auto').fit(altitude_data)
    sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center

# Plotting the Elbow curve
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.title("Elbow Method For Optimal k")
plt.show()

# %% [markdown]
# The number of cluster is 4 according to the elbow method
#

# %% [markdown]
# We used the sum of distances of samples to their closest cluster center in order to compute the metric for the Elbow curve

# %%
#TODO ecrire formule

# %%
kmeans = KMeans(n_clusters=4, random_state=0,n_init='auto')
merged_df['altitude_cluster'] = kmeans.fit_predict(altitude_data)


merged_df[['altitude','altitude_cluster']]

# %%
#TODO report finding

# %% [markdown]
#

# %%

# %% [markdown]
# ### b) **4/15** 
#
# Use `plotly` (or other similar graphing libraries) to create an interactive plot of the monthly median CO2 measurement for each site with respect to the altitude. 
#
# Add proper title and necessary hover information to each point, and give the same color to stations that belong to the same altitude cluster.

# %%

# %% [markdown]
# ### c) **6/15**
#
# Use `plotly` (or other similar graphing libraries) to plot an interactive time-varying density heatmap of the mean daily CO2 concentration for all the stations. Add proper title and necessary hover information.
#
# __Hints:__ Check following pages for more instructions:
# - [Animations](https://plotly.com/python/animations/)
# - [Density Heatmaps](https://plotly.com/python/mapbox-density-heatmaps/)

# %%


# %% [markdown]
# ## PART III: Model fitting for data curation (35 points)

# %% [markdown]
# ### a) **2/35**
#
# The domain experts in charge of these sensors report that one of the CO2 sensors `ZSBN` is exhibiting a drift on Oct. 24. Verify the drift by visualizing the CO2 concentration of the drifting sensor and compare it with some other sensors from the network. 

# %%

merged_df = merged_df.reset_index()

drifting_sensorDf = merged_df[merged_df['LocationName'] == 'ZSBN']

other_sensorsDf = merged_df[merged_df['LocationName'] != 'ZSBN']
other_sensorsDf = other_sensorsDf[["timestamp", "CO2"]]
other_sensorsDf = other_sensorsDf.groupby('timestamp').mean().reset_index()

# Plotting the CO2 concentration of the drifting sensor in comarison with the other sensors
fig = go.Figure()
fig.add_trace(go.Scatter(x=drifting_sensorDf['timestamp'], y=drifting_sensorDf['CO2'], name='Drifting Sensor (ZSBN)'))
fig.add_trace(go.Scatter(x=other_sensorsDf['timestamp'], y=other_sensorsDf['CO2'], name='Other Sensors', mode='markers'))
fig.update_layout(title='CO2 Concentration Comparison',
                  xaxis_title='Timestamp',
                  yaxis_title='CO2 Concentration')
#fig.show()



# %% [markdown]
# ### b) **8/35**
#
# The domain experts ask you if you could reconstruct the CO2 concentration of the drifting sensor had the drift not happened. You decide to:
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features the covariates not affected by the malfunction (such as temperature and humidity)
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __95% confidence interval__ obtained from cross validation: assume that the error follows a normal distribution and is independent of time.
# - What do you observe? Report your findings.
#
# __Note:__ Cross validation on time series is different from that on other kinds of datasets. The following diagram illustrates the series of training sets (in orange) and validation sets (in blue). For more on time series cross validation, there are a lot of interesting articles available online. scikit-learn provides a nice method [`sklearn.model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
#
# ![ts_cv](https://player.slideplayer.com/86/14062041/slides/slide_28.jpg)

# %%
# %%
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import bootstrap
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Assuming 'measurements' DataFrame is already loaded and available

# Select the dataset for training and the month for prediction, including 'timestamp'
X_before = merged_df.loc[(merged_df['LocationName'] == 'ZSBN') & (merged_df['timestamp'] < '2017-10-24'), ['temperature', 'humidity', 'timestamp']]
y_before = merged_df.loc[(merged_df['LocationName'] == 'ZSBN') & (merged_df['timestamp'] < '2017-10-24'), 'CO2']
X_month = merged_df.loc[(merged_df['LocationName'] == 'ZSBN'), ['temperature', 'humidity', 'timestamp']]
y_month = merged_df.loc[(merged_df['LocationName'] == 'ZSBN'), 'CO2']

# Define the columns to be used as features
cols_to_take = ['temperature', 'humidity']

# Filter columns for features, keeping 'timestamp' for later use
X_before = X_before.loc[:, cols_to_take + ['timestamp']]
X_month = X_month.loc[:, cols_to_take + ['timestamp']]

# Time Series Cross-Validation
reg = LinearRegression(n_jobs=-1)
splitter = TimeSeriesSplit(n_splits=5)  # Adjusted for simplicity

errors = []

for train_index, test_index in splitter.split(X_before[cols_to_take]):
    X_train, y_train = X_before.iloc[train_index][cols_to_take], y_before.iloc[train_index]
    X_test, y_test = X_before.iloc[test_index][cols_to_take], y_before.iloc[test_index]

    lin_reg = reg.fit(X_train, y_train)
    y_test_pred = lin_reg.predict(X_test)
    
    errors.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

mean_rmse = np.mean(errors)
ci = bootstrap((errors,), np.mean)

# Predicting for the entire month
lin_reg = LinearRegression(n_jobs=-1).fit(X_before[cols_to_take], y_before)
y_month_pred = lin_reg.predict(X_month[cols_to_take])

# Prepare dataframe for plotting
df_preds = pd.DataFrame({
    'timestamp': X_month['timestamp'],
    'CO2_pred': y_month_pred,
    'CO2_actual': y_month
})

df_preds['CI_high'] = df_preds['CO2_pred'] + (ci.confidence_interval.high - mean_rmse)
df_preds['CI_low'] = df_preds['CO2_pred'] - (mean_rmse - ci.confidence_interval.low)

# Plotting
fig = go.Figure()
fig.add_traces([
    go.Scatter(x=df_preds['timestamp'], y=df_preds['CI_low'], mode='lines', name="Lower 95% CI", line=dict(color='grey'), showlegend=False),
    go.Scatter(x=df_preds['timestamp'], y=df_preds['CI_high'], mode='lines', name='Upper 95% CI', line=dict(color='grey'), fill='tonexty', fillcolor='grey', showlegend=False),
    go.Scatter(name='True CO2 measurements', x=df_preds['timestamp'], y=df_preds['CO2_actual'], mode='lines+markers', line=dict(color='red')),
    go.Scatter(name='Predicted CO2 measurements', x=df_preds['timestamp'], y=df_preds['CO2_pred'], mode='lines+markers', line=dict(color='blue'))
])
fig.update_layout(title='CO2 Measurement Prediction with 95% Confidence Interval', xaxis_title='Date', yaxis_title='CO2 Concentration', legend_title='Legend')
#fig.show()

# Output the results
print(f"95% Confidence Interval for RMSE: [{ci.confidence_interval.low:.2f}, {ci.confidence_interval.high:.2f}]")
print(f"Average RMSE: {mean_rmse:.2f}")

# %% [markdown]
# findings :
#
#we see that the model recognizes the pattern of the co2 level. Meaning that the peaks are at the right time in general. The only problem is that the magnitude of the peaks are not exacly as the real ones. this could be due to other features that are not included in the inputs to the regression model.


# %% [markdown]
# ### c) **10/35**
#
# In your next attempt to solve the problem, you decide to exploit the fact that the CO2 concentrations, as measured by the sensors __experiencing similar conditions__, are expected to be similar.
#
# - Find the sensors sharing similar conditions with `ZSBN`. Explain your definition of "similar condition".
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features:
#     - the information of provided by similar sensors
#     - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __confidence interval__ obtained from cross validation
# - What do you observe? Report your findings.

# %%
#from geopy.distance import geodesic
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import bootstrap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from functools import reduce

# Assuming 'merged_df' and 'drifting_sensorDf' have been defined previously...

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the geodesic distance between two points."""
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Define similarity criteria and find similar sensors
distance_threshold = 5000  # meters
zsbn_info = merged_df[merged_df['LocationName'] == 'ZSBN'].iloc[0]
#similar_sensors = merged_df[merged_df.apply(lambda x: calculate_distance(zsbn_info['LAT'], zsbn_info['LON'], x['LAT'], x['LON']) <= distance_threshold and x['zone'] == zsbn_info['zone'] and x['LocationName'] != 'ZSBN', axis=1)]

# Prepare the dataset with separate features for each sensor
sensor_dfs = [drifting_sensorDf.rename(columns={'CO2': 'CO2_ZSBN', 'temperature': 'temperature_ZSBN', 'humidity': 'humidity_ZSBN'})[['timestamp', 'CO2_ZSBN', 'temperature_ZSBN', 'humidity_ZSBN']]]
# for sensor in similar_sensors['LocationName'].unique():
#     temp_df = merged_df[merged_df['LocationName'] == sensor][['timestamp', 'CO2', 'temperature', 'humidity']].rename(columns={'CO2': f'CO2_{sensor}', 'temperature': f'temperature_{sensor}', 'humidity': f'humidity_{sensor}'})
#     sensor_dfs.append(temp_df)

# Merge all sensor DataFrames on 'timestamp'
full_df = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), sensor_dfs)

# Drop any rows with NaN values that might have resulted from outer join
full_df.dropna(inplace=True)

# Define the model and perform cross-validation
X = full_df.drop(['timestamp', 'CO2_ZSBN'], axis=1)
y = full_df['CO2_ZSBN']

tss = TimeSeriesSplit(n_splits=5)
model = LinearRegression()
errors = []

for train_idx, test_idx in tss.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors.append(np.sqrt(mean_squared_error(y_test, y_pred)))

mean_rmse = np.mean(errors)
ci = bootstrap((errors,), np.mean, n_resamples=10000).confidence_interval

# Predicting for the entire dataset for plotting
y_pred_full = model.predict(X)

# Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(x=full_df['timestamp'], y=y, mode='markers', name='Actual CO2'))
fig.add_trace(go.Scatter(x=full_df['timestamp'], y=y_pred_full, mode='lines', name='Predicted CO2'))
fig.update_layout(title='CO2 Prediction vs Actual Measurements',
                  xaxis_title='Timestamp',
                  yaxis_title='CO2 Levels',
                  legend_title='Legend')
#fig.show()

# Output the results
print(f"Average RMSE: {mean_rmse:.2f}")
print(f"95% Confidence Interval for RMSE: [{ci.low:.2f}, {ci.high:.2f}]")

# %% 

# %% [markdown]
# ### d) **10/35**
#
# Now, instead of feeding the model with all features, you want to do something smarter by using linear regression with fewer features.
#
# - Start with the same sensors and features as in question c) 
# - Leverage at least two different feature selection methods -
# - Create similar interactive plot as in question c)
# - Describe the methods you choose and report your findings

# %% [code]
pivot_co2 = merged_df.pivot(index='timestamp', columns='LocationName', values='CO2')
pivot_temp = merged_df.pivot(index='timestamp', columns='LocationName', values='temperature')
pivot_humidity = merged_df.pivot(index='timestamp', columns='LocationName', values='humidity')

pivot_co2.columns = [f'CO2_{col}' for col in pivot_co2.columns]
pivot_temp.columns = [f'temp_{col}' for col in pivot_temp.columns]
pivot_humidity.columns = [f'humidity_{col}' for col in pivot_humidity.columns]

final_df = pd.concat([pivot_co2, pivot_temp, pivot_humidity], axis=1)

final_df.dropna(inplace=True)
y = final_df["CO2_ZSBN"]
X = final_df.drop(columns=["CO2_ZSBN"])

#%% 
merged_df 

# %% [code]
from sklearn.preprocessing import StandardScaler

X_preprocessed = X.drop(columns=['timestamp'], errors='ignore')
scaler = StandardScaler()
X_standardized = pd.DataFrame(scaler.fit_transform(X_preprocessed), columns=X_preprocessed.columns, index=X_preprocessed.index)

X_standardized

# %% [markdown]
# __Method 1: Select K Best features using F-statistic__

# %%
from sklearn.feature_selection import SelectKBest, f_regression

def select_features_selectkbest(X, y, k):
    """
    Selects the top k features using the SelectKBest method.

    Args:
    - X: DataFrame of features.
    - y: Target variable.
    - k: Number of top features to select.

    Returns:
    - X_selected: DataFrame with the selected features.
    - selector: Fitted SelectKBest object.
    """
    # Use F-statistic to select the features
    selector = SelectKBest(f_regression, k=k)
    
    # Fit the selector to the data. Note: Ensure X doesn't include non-numeric or timestamp columns
    X_new = X.drop(columns=['timestamp'], errors='ignore')  # Assuming 'timestamp' is in X and needs to be dropped
    selector.fit(X_new, y)
    
    # Transform the dataset to reduce to the selected features
    X_selected = selector.transform(X_new)
    
    # Get back to DataFrame to retain column names (optional)
    selected_features = X_new.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    # If you want to keep the timestamp column in the returned DataFrame,
    # assuming 'timestamp' was originally in X and was dropped before selection
    if 'timestamp' in X.columns:
        X_selected['timestamp'] = X['timestamp']
    
    X_selected = X_selected.reset_index(drop=True)
    
    return X_selected, selector

# %% [code]
# Select the k best features
X_selected = select_features_selectkbest(X_standardized, y, 5)
X_selected

# %% [markdown]
# __Method 2: Select Features Lasso__

# %% [code]
from sklearn.linear_model import LassoCV

def select_features_lasso(X, y, alpha=None):
    """
    Select features using LASSO (L1 regularization).
    
    Parameters:
    - X: DataFrame of features (ensure numeric and no timestamp).
    - y: Target variable.
    - alpha: The regularization strength; must be a positive float. 
             Smaller values specify stronger regularization. If None, 
             alpha is determined by cross-validation.
    
    Returns:
    - X_selected: DataFrame with the selected features based on non-zero coefficients.
    - model: Trained Lasso model.
    """
    X_numeric = X.drop(columns=['timestamp'], errors='ignore')
    
    if alpha is None:
        model = LassoCV(cv=5, random_state=42, max_iter=10000)
    else:
        model = Lasso(alpha=alpha)
    model.fit(X_numeric, y)
    
    # Get non-zero coefficients
    coef = pd.Series(model.coef_, index=X_numeric.columns)
    important_features = coef[coef != 0].index
    X_selected = X_numeric[important_features]
    
    return X_selected, model

# Example usage
X_selected_lasso, lasso_model = select_features_lasso(X, y)
X_selected_lasso



# %% [code]
X = X_selected_lasso
y = y
# %% [markdown]
# __Interactive Plot:__

# %% [code]
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import bootstrap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from functools import reduce

# Train using the selected features
tss = TimeSeriesSplit(n_splits=5)
model = LinearRegression()
errors = []

for train_idx, test_idx in tss.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors.append(np.sqrt(mean_squared_error(y_test, y_pred)))

mean_rmse = np.mean(errors)
ci = bootstrap((errors,), np.mean, n_resamples=10000).confidence_interval

# Predicting for the entire dataset for plotting
y_pred_full = model.predict(X)

# Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(x=full_df['timestamp'], y=y, mode='markers', name='Actual CO2'))
fig.add_trace(go.Scatter(x=full_df['timestamp'], y=y_pred_full, mode='lines', name='Predicted CO2'))
fig.update_layout(title='CO2 Prediction vs Actual Measurements',
                  xaxis_title='Timestamp',
                  yaxis_title='CO2 Levels',
                  legend_title='Legend')
fig.show()

# Output the results
print(f"Average RMSE: {mean_rmse:.2f}")
print(f"95% Confidence Interval for RMSE: [{ci.low:.2f}, {ci.high:.2f}]")

#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(X, y, n_components=None):
    """
    Apply PCA to reduce dimensionality of features.
    
    Parameters:
    - X: DataFrame of features (ensure numeric and no timestamp).
    - y: Target variable (not used directly by PCA but included for interface consistency).
    - n_components: Number of principal components to keep. If None, keep all components.
    
    Returns:
    - X_pca: DataFrame with the principal components as features.
    - pca: Fitted PCA object.
    """
    # Standardize the features before applying PCA
    X_numeric = X.drop(columns=['timestamp'], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    pca = PCA(n_components=n_components)
    X_pca_array = pca.fit_transform(X_scaled)
    
    # Create a DataFrame for the principal components
    pca_columns = [f'PC{i+1}' for i in range(X_pca_array.shape[1])]
    X_pca = pd.DataFrame(X_pca_array, columns=pca_columns, index=X.index)
    
    # Optionally, re-include the 'timestamp' column if needed

    X_pca['timestamp'] = X['timestamp']
    
    return X_pca, pca

# Example usage
n_components = 5  # Or another number, or None to keep all components
X_pca, pca_model = apply_pca(X, y, n_components=n_components)

# # You might want to examine how much variance is explained by the components
# explained_variance = pca_model.explained_variance_ratio_
# print(f"Explained variance by first {n_components} components:", explained_variance)

# %% [markdown]
# ### e) **5/35**
#
# Eventually, you'd like to try something new - __Bayesian Structural Time Series Modelling__ - to reconstruct counterfactual values, that is, what the CO2 measurements of the faulty sensor should have been, had the malfunction not happened on October 24. You will use:
# - the information of provided by similar sensors - the ones you identified in question c)
# - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
#
# To answer this question, you can choose between a Python port of the CausalImpact package (such as https://github.com/jamalsenouci/causalimpact) or the original R version (https://google.github.io/CausalImpact/CausalImpact.html) that you can run in your notebook via an R kernel (https://github.com/IRkernel/IRkernel).
#
# Before you start, watch first the [presentation](https://www.youtube.com/watch?v=GTgZfCltMm8) given by Kay Brodersen (one of the creators of the causal impact implementation in R), and this introductory [ipython notebook](https://github.com/jamalsenouci/causalimpact/blob/HEAD/GettingStarted.ipynb) with examples of how to use the python package.
#
# - Report your findings:
#     - Is the counterfactual reconstruction of CO2 measurements significantly different from the observed measurements?
#     - Can you try to explain the results?

# %% [code]
# %% [markdown]
# ### e) **5/35**
#
# Eventually, you'd like to try something new - __Bayesian Structural Time Series Modelling__ - to reconstruct counterfactual values, that is, what the CO2 measurements of the faulty sensor should have been, had the malfunction not happened on October 24. You will use:
# - the information of provided by similar sensors - the ones you identified in question c)
# - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
#
# To answer this question, you can choose between a Python port of the CausalImpact package (such as https://github.com/jamalsenouci/causalimpact) or the original R version (https://google.github.io/CausalImpact/CausalImpact.html) that you can run in your notebook via an R kernel (https://github.com/IRkernel/IRkernel).
#
# Before you start, watch first the [presentation](https://www.youtube.com/watch?v=GTgZfCltMm8) given by Kay Brodersen (one of the creators of the causal impact implementation in R), and this introductory [ipython notebook](https://github.com/jamalsenouci/causalimpact/blob/HEAD/GettingStarted.ipynb) with examples of how to use the python package.
#
# - Report your findings:
#     - Is the counterfactual reconstruction of CO2 measurements significantly different from the observed measurements?
#     - Can you try to explain the results?

#%% 
X.columns
#%% 
from causalimpact import CausalImpact
import pandas as pd

# Assuming ts_data is a DataFrame with the CO2 measurements from the ZSBN sensor
# and the covariates (temperature, humidity, etc.) from similar sensors.

# Define the pre-intervention period and the post-intervention period
# Example: Assuming the malfunction happened on October 24th, 2017
pre_period = ['2017-10-01', '2017-10-23']
post_period = ['2017-10-24', '2017-10-31']

# Run the causal impact analysis
#drop the timestamp column from X
X = X.drop(columns=['timestamp'], errors='ignore')

ci = CausalImpact(X, pre_period, post_period, nseasons=[{'period': 365}], model_args={"niter": 1000, "nseasons": 365})
print(ci.summary())
ci.plot()

# The summary output provides an overview of the estimated effect of the intervention
# The plot visualizes the observed data, the counterfactual predictions, and the pointwise differences

# %% [markdown]
#  To estimate the causal effect, we begin by specifying which period in the data should be used for training the model (pre-intervention period) and which period for computing a counterfactual prediction (post-intervention period).

# # %%
# from causalimpact import CausalImpact
# import pandas as pd

# # Assuming `data` is a DataFrame containing your target series and covariates
# # For example:
# # data['timestamp'] = pd.date_range(start='2023-10-01', end='2023-10-31', freq='D')
# # data['CO2_faulty'] = [your CO2 measurements for the faulty sensor]
# # data['temperature'] = [temperature measurements]
# # data['humidity'] = [humidity measurements]
# # data['CO2_similar'] = [CO2 measurements from similar sensors]

# # Set the pre-intervention period and the post-intervention period
# pre_period = ['2023-10-01', '2023-10-23']
# post_period = ['2023-10-24', '2023-10-31']

# # Drop the 'timestamp' column if it's not part of the analysis
# merged_df.drop(columns=['timestamp'], inplace=True, errors='ignore')

# # Run the causal impact analysis
# ci = CausalImpact(data, pre_period, post_period)
# print(ci.summary())
# ci.plot()


# %% [markdown]
#  This says that points for which the date is between the 1st and 23rd of October will be used for training, and time points for which the date is October 24th and after will be used for computing predictions. 
#  
#  Now, to perform inference, we need to run the analysis. But first, we need to define the data. The data we use here is the information provided by similar sensors.

# %%
# We define the "timestamp" column as the index of the initial data frame
covariates = X

 # %%
 # Finally, we define the data that we use to compute the predictions
# The data has a column for the CO2 measurements of the faulty sensor
# And temperature and humidity of all similar sensors (the covariates associated with the faulty sensors that were not affected by the malfunction)
ts_data = pd.concat([indexed_df[indexed_df['LocationName']=='ZSBN'][['CO2']], covariates], axis=1)

# %%
ts_impact = CausalImpact(ts_data, pre_period, post_period)
ts_impact.run()
ts_impact.plot()

# %%
ts_impact.summary() 

# %% [markdown]
# # That's all, folks!

# %% [markdown]
# # That's all, folks!

# %%