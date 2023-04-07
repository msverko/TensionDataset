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
from scipy.stats import shapiro
from scipy.stats import anderson

df = pd.read_csv(r"GitTest\DVL2_DATASET_MIN_FULL_2023-04-06.csv", sep=';', encoding='utf-8')
df['Time'] = pd.to_datetime(df['Time']) #first df column (Time) was initially interpreted as object type
df.set_index('Time', inplace=True) #time column used as index


print("\n ########### Dataframe info ##########")
dfinfo = df.info(verbose=True, show_counts=True)

print("\n ########### Describe dataframe ##########")
summary_stats = df.describe()
summary_stats.to_csv('df_summary.csv')
print(summary_stats)


# --------- Plot some of the variables - START
vars_to_plot = ['68_0', '68_4', '60_69', '60_69', '69_10', '69_36', '69_49', '69_62', '69_75', '69_88', '69_101']
ax = df[vars_to_plot].plot(figsize=(16,7))
ax.set_title('Speed mode in relation to Coil diameters')
ax.set_xlabel('Time')
ax.set_ylabel('Value [%]')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
plt.subplots_adjust(right=0.8)
plt.show()



'''
# normalize the dataframe excluding the index column
df_no_index = df.reset_index(drop=True)
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df_no_index)
df_normalized = pd.DataFrame(scaled_values, columns=df_no_index.columns)
# merge the index column back onto the normalized dataframe
df_normalized['time'] = df.index
df_normalized.set_index('time', inplace=True)
'''

test_results = pd.DataFrame(columns=['Column', 'Statistic', 'p-value'])
# loop through columns and perform Shapiro-Wilk test on each column
for column in df.columns:
    stat, p = shapiro(df[column])
    test_results = test_results.append({'Column': column, 'Statistic': stat, 'p-value': p}, ignore_index=True)
test_results.to_csv(r'GitTest\Normality_test_results_Saphiro.csv', index=False)

test_results = pd.DataFrame(columns=['Column', 'Statistic', 'Critical Values', 'Significance Levels'])
# loop through columns and perform Anderson-Darling test on each column
for column in df.columns:
    result = anderson(df[column])
    test_results = test_results.append({'Column': column, 'Statistic': result.statistic,
                                        'Critical Values': result.critical_values,
                                        'Significance Levels': result.significance_level},
                                        ignore_index=True)
test_results.to_csv(r'GitTest\Normality_tesr_results_Anderson.csv', index=False)



# Calculate statistics for each column
stats = df.describe(percentiles=[0.25, 0.5, 0.75])
means = stats.loc['mean']
stds = stats.loc['std']
vars = stats.loc['std']**2
p25s = stats.loc['25%']
p50s = stats.loc['50%']
p75s = stats.loc['75%']

# --------- Distribution with marked statistics for every variable in df ploted in 3x4 grid - START
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
    
    # Plot the distribution - START
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
# --------- Distribution with marked statistics for every variable in df ploted in 3x4 grid - END

# --------- Boxplot(normalized dataframe) - START
sns.boxplot(data=df)
plt.xticks(rotation=90)
ax = plt.gca()  # Get the current Axes instance
plt.subplots_adjust(bottom=0.31, top=0.95)  # Adjust the bottom margin
plt.show()
# --------- Boxplot - END

# --------- Heatmap - START
corr_matrix = df.corr()
sns.set(font_scale=0.6)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.05f', ax=ax)
plt.yticks(rotation=0)
fig.tight_layout(rect=[0, 0.1, 1, 0.9]) # Heatmap position
plt.show()
# --------- Heatmap - END



