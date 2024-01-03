# Generating, plotting, and saving the data for K-means clustering

import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(0)

# Parameters: number of clusters and data points
k = 3
n_points = 300

# Generate random centroids within a 100x100 area
centers = np.random.rand(k, 2) * 100

# Generate points around each centroid with Gaussian distribution
points = []
for c in centers:
    points.append(c + np.random.randn(n_points // k, 2) * 10)
points = np.vstack(points)

# Plotting the generated data
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], s=10)
plt.title("Generated Data for K-Means Clustering")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)

# Save the plot to an output file
plot_path = '/mnt/data/k_means_data_plot.png'
plt.savefig(plot_path)

# Save the generated points to a file
data_path = '/mnt/data/k_means_data.txt'
with open(data_path, 'w') as file:
    for point in points:
        file.write(f"{point[0]} {point[1]}\n")

plot_path, data_path

