import sys
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


print("----- Python ver." + sys.version)

df = pd.read_csv("DVL2_DATASET_MIN_2023-04-03.csv", sep=';', encoding='utf-8')
df['time'] = pd.to_datetime(df['time']) #first df column (time) was initially interpreted as object type
df.set_index('time', inplace=True) #time column used as index

print("\n ########### Dataframe info ##########")
df.info(verbose=True, show_counts=True)

print("\n ########### Dataframe head ##########")
print(df.head())

print("\n ########### Dataframe tail ##########")
print(df.tail())

print("\n ########### Count null variables (NaN) ##########")
print(df.isna().sum())

print("\n ########### Describe dataframe ##########")
print(df.describe())

print("\n ########### Corelations ##########")
print(df.corr())


#df['TRK_HMI_STS_DB\DATA.TRK_WS_EN1_COIL_DIAM'].plot(logy=True)
#sns.heatmap(df.corr(), annot=True)
#plt.show()

#First drop target column, then include only target column
x_train, x_test, y_train, y_test = train_test_split(df.drop('LNEN_DRV_WS_BRD1_1_SPD_SET',
                                                    axis=1),
                                                    df['LNEN_DRV_WS_BRD1_1_SPD_SET'],
                                                    random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.score(x_train, y_train))
print(lr.score(x_test, y_test))

print(lr.coef_)
print(lr.intercept_)

sns.barplot(x=lr.coef_, y=x_train.columns, color='darkblue')
plt.show()