
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit

# Load the dataset
data = 'dataset.csv'
data_df = pd.read_csv(data_file_path, skiprows=4)

# Function for linear curve fitting
def linear_fit(x, a, b):
    return a * x + b

# Filter data for CO2 emissions (metric tons per capita)
co2_emissions_data = data_df[data_df['Indicator Code'] == 'EN.ATM.CO2E.PC']

# Calculating the global average of CO2 emissions per year
global_co2_average = co2_emissions_data.mean()

# Preparing data for KMeans Clustering
# Taking years as features for clustering
years = np.array([int(year) for year in global_co2_average.index if year.isnumeric()])
co2_values = global_co2_average.values[:len(years)]

# Reshaping data for clustering
cluster_data = np.vstack((years, co2_values)).T

# KMeans Clustering
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(cluster_data)

# Plotting the Global Trend Plot for CO2 Emissions with Clustering
plt.figure(figsize=(15, 7))
sns.scatterplot(x=years, y=co2_values, hue=clusters, palette='viridis')
plt.title('Global CO2 Emissions Trend with KMeans Clustering (1960-2021)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (Metric Tons Per Capita)')
plt.grid(True)
plt.show()

# Curve Fitting for Global CO2 Emissions Trend
# Fit the data to the linear curve
params, params_covariance = curve_fit(linear_fit, years, co2_values)

# Plotting the curve fit
plt.figure(figsize=(15, 7))
plt.scatter(years, co2_values, label='Data')
plt.plot(years, linear_fit(years, *params), label='Fitted function', color='red')
plt.title('Curve Fitting for Global CO2 Emissions Trend')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (Metric Tons Per Capita)')
plt.legend()
plt.grid(True)
plt.show()
