import pandas as pd
import sklearn as sk
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.graphics.api as smg
import plotly.express as px
from datetime import datetime
from scipy.spatial.distance import mahalanobis
import numpy as np
from scipy.spatial import distance as dist

#date parser setup
dateparse = lambda x: datetime.strptime(x, '%Y%m%d')
dateparse2 = lambda x: datetime.strptime(x, '%Y')

#reading the file
df = pd.read_csv('Rent_S1_Done.csv', parse_dates=['g_day'], date_parser=dateparse)


#Adding price per square meter to evaluate ratio realism
df['psqm'] = df.apply(lambda row: row.a_netm_mon / row.a_surface_living, axis=1)

print(1)



#fig = px.histogram(df_cleaned, x="a_netm_mon", marginal="box")
#fig.show()


#------#------#------#------#------#------#------#------#------#------#------





#Drop usless variables
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('deal', axis=1)
#df = df.drop('a_nb_rooms', axis=1)

print(2)

df['transac_year'] = pd.DatetimeIndex(df['g_day']).year
df['age'] = df.apply(lambda row: row.transac_year - row.a_baup, axis=1)
print(3)

df = df[df.a_netm_mon > 1]
df = df[df.psqm < 170]
df = df[df.psqm > 8]
df = df[df.a_nb_rooms < 30]
df = df[df.a_baup < 2015]
df = df[df.a_baup > 1]
df = df[df.age < 700]
df = df[df['a_kat_o_2'].isin(['AP', 'HO'])]
df.drop(df.loc[df['a_nb_rooms']==1.04].index, inplace=True)
df.drop(df.loc[df['a_nb_rooms']==1.07].index, inplace=True)
df.drop(df.loc[df['a_nb_rooms']==0.5].index, inplace=True)
df.drop(df.loc[df['a_sicht']== -1].index, inplace=True)
df.drop(df.loc[df['a_baup']== 9999].index, inplace=True)
df.drop(df.loc[df['a_balkon']==-1].index, inplace=True) #unkown entry code

print(4)
#df.loc[df['a_sicht'] == -1, 'a_sicht'] = 0
#df.loc[df['a_balkon'] == -1, 'a_balkon'] = 0
#df.loc[df['a_ofen'] == -1, 'a_ofen'] = 0


print(df.columns)
print(df['a_kat_o_2'].value_counts())


#------#------#------#------#------#------#------#------#------#------#------

# Select variables for testing multivariate outliers
variables = ['a_netm_mon', 'a_surface_living', 'age']
data = df[variables]

# Calculate the mean and covariance matrix of the data
mean = data.mean()
covariance = data.cov()

# Calculate the Mahalanobis distance
distances = []
for index, row in data.iterrows():
    distance = dist.mahalanobis(row, mean, np.linalg.inv(covariance))
    distances.append(distance)

# Set a threshold and identify outliers with a greater distance
threshold = np.percentile(distances, 95)

outliers = []
for i, distance in enumerate(distances):
    if distance > threshold:
        outliers.append(i)

# Calculate the significance level (5%)
significance_level = 0.05

# Estimate the expected number of false positives
expected_false_positives = int(significance_level * len(distances))


# Remove outliers from the dataframe
df_cleaned = df.drop(df.index[outliers])

# Print the number of outliers removed
print('Number of outliers removed:', len(outliers))

df = df_cleaned

print(5)

#------#------#------#------#------#------#------#------#------#------


#Remove data points outside of Switzerland

# Define the bounding box for Switzerland
switzerland_bbox = [45.817, 5.955, 47.808, 10.492]

# Filter the dataframe to include only rows within the bounding box
df = df[(df['latitude2'].between(switzerland_bbox[0], switzerland_bbox[2])) & (df['longitude2'].between(switzerland_bbox[1], switzerland_bbox[3]))]

# Group the dataframe by latitude and longitude to count the number of entries in each location
grouped = df.groupby(['latitude2', 'longitude2']).size().reset_index(name='count')

# Display the first few rows of the grouped dataframe
print(grouped.head())


df.to_csv("Rent_S2_Done.csv")
