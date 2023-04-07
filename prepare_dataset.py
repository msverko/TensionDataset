import sys
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_csv("GitTest\DVL2_DATASET_MIN_2023-04-06.csv", sep=';', encoding='utf-8')
df1['Time'] = pd.to_datetime(df1['Time']) #first df column (time) was initially interpreted as object type
# !!! Attention if tagnames column is used as varnames, above column 'Time' is with lower case 'time' in csv file
df1.set_index('Time', inplace=True) #time column used as index

df2 = pd.read_csv("GitTest\DVL2_DATASET_MIN_2023-04-06_DRV_SPD_MODE.csv", sep=';', encoding='utf-8')
df2['Time'] = pd.to_datetime(df2['Time']) #first df column (time) was initially interpreted as object type
# !!! Attention if tagnames column is used as varnames, above column 'Time' is with lower case 'time' in csv file
df2.set_index('Time', inplace=True) #time column used as index

df3 = pd.read_csv("GitTest\DVL2_DATASET_MIN_2023-04-06_StripTck.csv", sep=';', encoding='utf-8')
df3['Time'] = pd.to_datetime(df3['Time']) #first df column (time) was initially interpreted as object type
# !!! Attention if tagnames column is used as varnames, above column 'Time' is with lower case 'time' in csv file
df3.set_index('Time', inplace=True) #time column used as index

# concatenate the two datasets
df = pd.concat([df1, df2, df3], axis=1)
df.to_csv('GitTest\DVL2_DATASET_MIN_FULL_2023-04-06.csv', sep=';', encoding='utf-8', index=True)

print("\n ########### Dataframe info ##########")
dfinfo = df.info(verbose=True, show_counts=True)