'''
Vehicle Price Prediction Project
By: Mahen Muthukumarana
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from scipy.stats import f_oneway # in numpy

# Load the dataset
vehicles = pd.read_csv('vehicle_price_prediction.csv')
print(vehicles.head())


# ensuring data is accurate/needs cleaning
print(vehicles.describe())
print(vehicles.describe().iloc[:,3:5])
print("\n\n")
print(vehicles.isna().any())

# replacing None with 'None' string in accident_history column. Now has 3 unique values instead of 2
print("\n\n")
print(vehicles['accident_history'].nunique())
print(vehicles['accident_history'].value_counts())

print("\n\n")
vehicles['accident_history'] = vehicles['accident_history'].fillna('None')
print(vehicles.isna().any())
print("\n\n")
print(vehicles['accident_history'].nunique())
print(vehicles['accident_history'].value_counts())

print("\n\n")
print(vehicles.dtypes)
print("\n\n")
print(vehicles.mode())


print("ALL COLUMN NAMES: ", vehicles.columns)
#quantitative variables
print("\nALL NUMERICAL COLUMN NAMES: ", vehicles.describe().columns)


# EDA (exploratory data analysis)

''' measures of central tendancy
mean > median > mode = right skewness
mean < median < mode = left skewness
'''



quantitative_variables = ['year', 'mileage', 'engine_hp', 'owner_count', 'vehicle_age',
       'mileage_per_year', 'brand_popularity', 'price']

vehicles_quantitative = vehicles.select_dtypes(include=['int64', 'float64'])
print(vehicles_quantitative)

# Creating multiple subplots of histograms for all quantitative variables in vehicles
fig, axs = plt.subplots(3,3)
axs[0][0].hist(vehicles['price']) # right skewed with a large frequency for min (1500)
plt.xlabel("price")
axs[0][1].hist(vehicles['year']) # left skewed
plt.xlabel("year")
axs[0][2].hist(vehicles['mileage']) 
plt.xlabel("mileage")
axs[1][0].hist(vehicles['vehicle_age']) 
plt.xlabel("vehicle_age")
axs[1][1].hist(vehicles['engine_hp']) 
plt.xlabel("engine_hp")
axs[1][2].hist(vehicles['mileage_per_year']) 
plt.xlabel("mileage_per-year")
axs[2][0].hist(vehicles['owner_count'])
plt.xlabel("owner_count")
axs[2][1].hist(vehicles['brand_popularity']) 
plt.xlabel("brand_popularity")
plt.tight_layout()
plt.show() 
plt.clf()


sns.histplot(data = vehicles, x = 'price') 
plt.show() 
plt.clf()
sns.histplot(data = vehicles, x = 'year') 
plt.tight_layout()
plt.show() 
plt.clf()


sns.scatterplot(x='year', y='vehicle_age', data = vehicles) # perfect negative correlation between year and vehicle age, linear relationship
plt.show()
plt.clf()

# plot of all pairs of variables
sns.pairplot(vehicles_quantitative)
plt.show()

# heatmap plot
#correlation_matrix = vehicles_quantitative.corr()
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.title('Correlation Heatmap')
#plt.show()


correlation = vehicles['year'].corr(vehicles['vehicle_age'])
print(f"\n\nCorrelation between Variable 1 and Variable 2: {correlation}")



contingency_table = pd.crosstab(vehicles['make'], vehicles['drivetrain'])
print(contingency_table)

group1 = vehicles[vehicles['make'] == 'Tesla']['mileage_per_year']
group2 = vehicles[vehicles['make'] == 'Porsche']['mileage_per_year']
f_statistic, p_value = f_oneway(group1, group2)
print(f"F-statistic: {f_statistic}, P-value: {p_value}")
print(group1)
print('\n\n')
print(group2)


print("\nmean of Tesla's mileage per year: ", group1.mean())
print("mean of Porsche's mileage per year: ", group2.mean())


# ML MODEL CREATION
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


vehicles_engine_hp_and_price = vehicles[['engine_hp','price']] # both right-skewed (by frequency) numerical variables. see plots above
print(vehicles_engine_hp_and_price)
X = np.array(vehicles_engine_hp_and_price['engine_hp']).reshape(-1,1)
y = np.array(vehicles_engine_hp_and_price['price']).reshape(-1,1)

# PARAMETER: test size=.2 means 20% testing, 80% training
# PARAMETER: random_state = 42 ensures reproducibility in the data split in training and testing sets every time command runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42) 
regr = LinearRegression()
regr.fit(X_train, y_train)
r_2_10_90 = regr.score(X_test, y_test)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=.2, random_state=42) 
regr = LinearRegression()
regr.fit(X_train1, y_train1)
r_2_20_80 = regr.score(X_test1, y_test1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=.3, random_state=42) 
regr = LinearRegression()
regr.fit(X_train2, y_train2)
r_2_30_70 = regr.score(X_test2, y_test2)

# R^2 value is the regression score or correlation squared. there is no universal "good" R^2 value. 1 is overfitting isn't necessarily perfect in real applications. 0 is no better than calculating the mean.
print("\n\nR^2 value (10% testing and 90% training):", r_2_10_90)
print("R^2 value: (20% testing and 80% training)", r_2_20_80)
print("R^2 value: (30% testing and 70% training)", r_2_30_70)
"""
test_size =.8 -> r_2 = 0.4270258091043161
test_size =.7 -> r_2 = 0.42678064194027476
test_size =.6 -> r_2 = 0.4267739687768145
test_size =.5 -> r_2 =  0.42670267706214715
test_size =.4 -> r_2 = 0.42644738868042775
test_size =.3 -> r_2 =  0.42733976062410195
test_size =.2 -> r_2 = 0.4272738815349667
test_size =.15 -> r_2 = 0.4283697509498735
test_size =.1 -> r_2 =  0.4280669156206006
test_size =.05 -> r_2 = 0.42660597704277836
test_size =.01 -> r_2 = R^2 value: 0.42711208686756463
"""

y_predict = regr.predict(X_test)
plt.scatter(X_test, y_test, color = 'b')
plt.plot(X_test, y_predict, color= 'r')
plt.show()
plt.clf()

y_predict1 = regr.predict(X_test1)
plt.scatter(X_test1, y_test1, color = 'b')
plt.plot(X_test1, y_predict1, color= 'r')
plt.show()
plt.clf()

y_predict2 = regr.predict(X_test2)
plt.scatter(X_test2, y_test2, color = 'b')
plt.plot(X_test2, y_predict2, color= 'r')
plt.show()
plt.clf()


#plotting the Scatter plot to check relationship between Sal and Temp 
'''sns.scatterplot(x ="engine_hp", y ="price", data = vehicles)
sns.lmplot(x ="engine_hp", y ="price", data = vehicles, order = 2, ci = None)
plt.tight_layout()
plt.show()'''




