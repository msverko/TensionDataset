import sys
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import anderson


df = pd.read_csv(r"GitTest\DVL2_DATASET_MIN_FULL_2023-04-06.csv", sep=';', encoding='utf-8')
df['Time'] = pd.to_datetime(df['Time']) #first df column (Time) was initially interpreted as object type
df.set_index('Time', inplace=True) #time column used as index

# Perform the Box-Cox transformation
#transformed_data, lambda_value = stats.boxcox(df['13_1'])

# Perform the Yeo-Johnson transformation (more flexible than Box-Cox; it can be performed on negative values)
transformed_data, lambda_value = stats.yeojohnson(df['13_1'])

# Print the lambda value, which is the optimal power parameter for the transformation (-1 to 1 = weak trasnformation)
print(lambda_value)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].hist(df['13_1'])
axs[0].set_title("Original Data")

axs[1].hist(transformed_data)
axs[1].set_title("Transformed Data")
plt.show()


sns.boxplot(data=df['13_1'])
ax = plt.gca()  # Get the current Axes instance
plt.subplots_adjust(bottom=0.31, top=0.95)  # Adjust the bottom margin
plt.show()
