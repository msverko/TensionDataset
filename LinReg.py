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

df = pd.read_csv("GitTest\DVL2_DATASET_MIN_2023-04-03.csv", sep=';', encoding='utf-8')
df['time'] = pd.to_datetime(df['time']) #first df column (time) was initially interpreted as object type
df.set_index('time', inplace=True) #time column used as index

#Normalize dataset
scaler = MinMaxScaler(feature_range=(0, 100))
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

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

mean = np.mean(df)
median = np.median(df)
std_dev = np.std(df)
percentiles = np.percentile(df, [25, 50, 75])
variance = np.var(df)


print('\n Mean:\n', mean)
print(' Percentiles:\n', percentiles)
print(' Standard deviation:\n', std_dev)
print(' Variance:\n', variance)
print(" Distribution:\n", distribution) # Returns two arrays (first = bins frequencies, second= bin edges)



sns.boxplot(data=df)
plt.xticks(rotation=90)
ax = plt.gca()  # Get the current Axes instance
plt.subplots_adjust(bottom=0.31, top=0.95)  # Adjust the bottom margin
plt.show()

'''
# Create histogram plot for each variable in DataFrame
sns.histplot(data=df_normalized, kde=True)
# Add legend box
plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Histogram of the entire dataframe (normalized)')
plt.show()
'''



corr_matrix = df.corr()
# Increase font size
sns.set(font_scale=0.6)
# Create figure and axis objects
fig, ax = plt.subplots(figsize=(12, 10))
# Create heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.05f', ax=ax)
# Rotate y-axis labels
plt.yticks(rotation=0)
# Adjust position of heatmap within figure
fig.tight_layout(rect=[0, 0.1, 1, 0.9])
# Show plot
plt.show()



#sns.histplot(data=df, x='LN_PR_FURN_EN_LDCL_TENS_ACT', kde=True)
#plt.show()


#df['TRK_HMI_STS_DB\DATA.TRK_WS_EN1_COIL_DIAM'].plot(logy=True)
#sns.heatmap(df.corr(), annot=True)
#plt.show()

#First drop target column, then include only target column

'''
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

