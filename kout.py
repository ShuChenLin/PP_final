# The user has uploaded another file. I will read this file and create a scatter plot based on its contents.

# Updating the file path for the newly uploaded file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
file_path = 'kdata.out'

# Attempting to read the new file and create a scatter plot
try:
    # Reading the new file
    df_latest = pd.read_csv(file_path, delim_whitespace=True)
    
    # Creating a scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_latest, x='x', y='y', hue='cluster', palette='bright', s=50)  # Increased marker size for better visibility
    plt.title('Latest Cluster Scatter Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Showing the plot
    plot_ready = True
except Exception as e:
    plot_ready = False
    error_message = str(e)

plot_ready, error_message if not plot_ready else "Plot created successfully."

