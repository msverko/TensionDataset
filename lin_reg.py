import sys
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

#print("----- Python ver." + sys.version)

df = pd.read_csv("GitTest\DVL2_DATASET_MIN_2023-04-05.csv", sep=';', encoding='utf-8')
df['time'] = pd.to_datetime(df['time']) #first df column (time) was initially interpreted as object type
df.set_index('time', inplace=True) #time column used as index


#df.columns = df.columns.str.replace('TRK_HMI_STS_DB\DATA.TRK', 'TRK')
#df.rename(columns={'DB 15200.DBD 1538': 'TRK_STRIP_THICKNESS'}, inplace=True)

'''
#Normalize dataset
scaler = MinMaxScaler(feature_range=(0, 100))
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
'''

print("\n ########### Dataframe info ##########")
dfinfo = df.info(verbose=True, show_counts=True)
with open("df_info.csv", "w") as f:
    f.write(str(dfinfo))

print("\n ########### Count null variables (NaN) ##########")
print(df.isna().sum())

print("\n ########### Describe dataframe ##########")
summary_stats = df.describe()
summary_stats.to_csv('df_summary.csv')
print(summary_stats)

print("\n ########### Corelations ##########")
print(df.corr())

distribution = np.histogram(df)
print("\n\n Distribution:\n", distribution)




'''
# Create histogram plot for each variable in DataFrame
sns.histplot(data=df_normalized, kde=True)
# Add legend box
plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Histogram of the entire dataframe (normalized)')
plt.show()
'''

'''
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
'''
#sns.barplot(x=lr.coef_, y=x_train.columns, color='darkblue')
#plt.show()

