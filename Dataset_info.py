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


df = pd.read_csv("GitTest\DVL2_DATASET_MIN_2023-04-03.csv", sep=';', encoding='utf-8')
df['time'] = pd.to_datetime(df['time']) #first df column (time) was initially interpreted as object type
df.set_index('time', inplace=True) #time column used as index

# Calculate statistics for each column
stats = df.describe(percentiles=[0.25, 0.5, 0.75])
means = stats.loc['mean']
stds = stats.loc['std']
vars = stats.loc['std']**2
p25s = stats.loc['25%']
p50s = stats.loc['50%']
p75s = stats.loc['75%']

#Number of plots created
plot_count = 0

legend_labels = ['Mean', 'Std.Dev.', 'Std.Dev.', 'Q1', 'Q2', 'Q3']
legend_colors = ['red', 'purple', 'purple', 'lightgreen', 'green', 'darkgreen']

# Loop through each column and plot its distribution
for i, col in enumerate(df.columns):
    # Create a new 3x4 grid every 12 plots
    if i % 12 == 0:
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
        axes = axes.flatten()
        plot_count = 0
    
    # Plot the distribution
    df[col].hist(ax=axes[plot_count], bins=20)
    axes[plot_count].set_title(col)
    axes[plot_count].axvline(means[i], color='red', linestyle='dashed', linewidth=2, label='Mean')
    axes[plot_count].axvline(means[i]-stds[i], color='purple', linestyle='dashed', linewidth=2, label='Std.Dev.')
    axes[plot_count].axvline(means[i]+stds[i], color='purple', linestyle='dashed', linewidth=2, label='Std.Dev.')
    axes[plot_count].axvline(p25s[i], color='lightgreen', linestyle='dashed', linewidth=2, label='Q1')
    axes[plot_count].axvline(p50s[i], color='green', linestyle='dashed', linewidth=2, label='Q1')
    axes[plot_count].axvline(p75s[i], color='darkgreen', linestyle='dashed', linewidth=2, label='Q1')
    axes[plot_count].text(0.95, 0.95, f'mean: {means[i]:.2f}\nstd: {stds[i]:.2f}\nvar: {vars[i]:.2f}', 
                           ha='right', va='top', transform=axes[plot_count].transAxes)
    
    # Add a legend to the plot
    legend_handles = [axes[plot_count].axvline(0, color=color, linestyle='--') for color in legend_colors]
    axes[plot_count].legend(legend_handles, legend_labels, loc='upper left')

    # Increment the plot count
    plot_count += 1
    
    # Hide unused axes on the last plot
    if i == len(df.columns) - 1:
        for j in range(plot_count, len(axes)):
            axes[j].axis('off')

    # Show the plot if a full grid has been created or if it's the last plot
    if plot_count == 12 or i == len(df.columns) - 1:
        plt.tight_layout()
        plt.show()