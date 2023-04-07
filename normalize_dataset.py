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
print(df.index.values)

df1 = pd.DataFrame(index=df.index)

# Perform the Box-Cox transformation
#transformed_data, lambda_value = stats.boxcox(df['13_1'])

# Perform the Yeo-Johnson transformation (more flexible than Box-Cox; it can be performed on negative values)
transformed_data, lambda_value = stats.yeojohnson(df['13_1'])
df1['13_1t'] = transformed_data


# Print the lambda value, which is the optimal power parameter for the transformation (-1 to 1 = weak trasnformation)
print(lambda_value)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].hist(df['13_1'])
axs[0].set_title("Original Data")

axs[1].hist(transformed_data)
axs[1].set_title("Transformed Data")
plt.show()

vars_to_plot = [df['13_1'],transformed_data]
#sns.boxplot(data=df['13_1', transformed_data])
sns.boxplot(vars_to_plot)
ax = plt.gca()  # Get the current Axes instance
plt.subplots_adjust(bottom=0.31, top=0.95)  # Adjust the bottom margin
plt.show()


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(df['13_1'], color='blue')
axs[0].set_title("Original Data")

axs[1].plot(df1['13_1t'], color='red')
axs[1].set_title("Transformed Data")

plt.show()



df1 = df.apply(lambda x: stats.yeojohnson(x)[0])


sta1 = print(df.describe().transpose()[['min', 'max']])
sta1.to_csv('sta1.csv')
sta2 = print(df1.describe().transpose()[['min', 'max']])
sta2.to_csv('sta2.csv')

# Calculate statistics for each column of transformed df1
stats = df1.describe(percentiles=[0.25, 0.5, 0.75])
means = stats.loc['mean']
stds = stats.loc['std']
vars = stats.loc['std']**2
p25s = stats.loc['25%']
p50s = stats.loc['50%']
p75s = stats.loc['75%']

legend_labels = ['Mean', 'Std.Dev.', 'Std.Dev.', 'Q1', 'Q2', 'Q3']
legend_colors = ['red', 'purple', 'purple', 'lightgreen', 'green', 'darkgreen']


for i, col in enumerate(df.columns):
    # Create a new 4x2 grid every 8 plots
    if i % 4 == 0:
        fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))
        axs = axs.flatten()
        plot_count = 0
    print(str(i))

    # iterate over the columns of each DataFrame and plot histograms
    axs[plot_count].hist(df[col])
    axs[plot_count].set_title(col + ' (df)')
    axs[plot_count].axvline(means[i], color='red', linestyle='dashed', linewidth=2, label='Mean')
    axs[plot_count].axvline(means[i]-stds[i], color='purple', linestyle='dashed', linewidth=2, label='Std.Dev.')
    axs[plot_count].axvline(means[i]+stds[i], color='purple', linestyle='dashed', linewidth=2, label='Std.Dev.')
    axs[plot_count].axvline(p25s[i], color='lightgreen', linestyle='dashed', linewidth=2, label='Q1')
    axs[plot_count].axvline(p50s[i], color='green', linestyle='dashed', linewidth=2, label='Q1')
    axs[plot_count].axvline(p75s[i], color='darkgreen', linestyle='dashed', linewidth=2, label='Q1')
    axs[plot_count].text(0.95, 0.95, f'mean: {means[i]:.2f}\nstd: {stds[i]:.2f}\nvar: {vars[i]:.2f}', 
                           ha='right', va='top', transform=axs[plot_count].transAxes)

    # Add a legend to the plot
    legend_handles = [axs[plot_count].axvline(0, color=color, linestyle='--') for color in legend_colors]
    axs[plot_count].legend(legend_handles, legend_labels, loc='upper left')

    # iterate over the columns of each DataFrame and plot histograms
    axs[plot_count + 1].hist(df1[col])
    axs[plot_count + 1].set_title(col + ' (df1)')
    axs[plot_count + 1].axvline(means[i], color='red', linestyle='dashed', linewidth=2, label='Mean')
    axs[plot_count + 1].axvline(means[i]-stds[i], color='purple', linestyle='dashed', linewidth=2, label='Std.Dev.')
    axs[plot_count + 1].axvline(means[i]+stds[i], color='purple', linestyle='dashed', linewidth=2, label='Std.Dev.')
    axs[plot_count + 1].axvline(p25s[i], color='lightgreen', linestyle='dashed', linewidth=2, label='Q1')
    axs[plot_count + 1].axvline(p50s[i], color='green', linestyle='dashed', linewidth=2, label='Q1')
    axs[plot_count + 1].axvline(p75s[i], color='darkgreen', linestyle='dashed', linewidth=2, label='Q1')
    axs[plot_count + 1].text(0.95, 0.95, f'mean: {means[i]:.2f}\nstd: {stds[i]:.2f}\nvar: {vars[i]:.2f}', 
                           ha='right', va='top', transform=axs[plot_count].transAxes)

    # Add a legend to the plot
    legend_handles = [axs[plot_count + 1].axvline(0, color=color, linestyle='--') for color in legend_colors]
    axs[plot_count + 1].legend(legend_handles, legend_labels, loc='upper left')

    # Increment the plot count
    plot_count += 2 
    print('Plot cnt: ' + str(plot_count))
        
    # Hide unused axes on the last plot
    if i == len(df.columns) - 1:
        for j in range(plot_count, len(axs)):
            axs[j].axis('off')

    # Show the plot if a full grid has been created or if it's the last plot
    if plot_count == 8 or i == len(df.columns) - 1:
        plt.tight_layout()
        plt.show()
        

