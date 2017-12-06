'''
Lodewijk Brand's Clustering Assignment using Tensorflow

Homework #2, Advanced High-Performance Computing
'''

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

DIM=3

parser = argparse.ArgumentParser(description='Lets try clustering in Tensorflow!')
parser.add_argument('--num_points', type=int, help='Number of points to cluster')
parser.add_argument('--num_clusters', type=int, help='Number of clusters')
parser.add_argument('--num_iterations', type=int, help='Number of iterations')

args = parser.parse_args()

if(args.num_points):
    NUM_SAMPLES = args.num_points
else:
    NUM_SAMPLES = 100
    print("--num_points set to a default value: ", str(NUM_SAMPLES)) 

if(args.num_clusters):
    NUM_CLUSTERS = args.num_clusters
else:
    NUM_CLUSTERS = 4
    print("--num_clusters set to a default value: " + str(NUM_CLUSTERS)) 

if(args.num_iterations):
    ITERATIONS = args.num_iterations
else:
    ITERATIONS = 1000
    print("--num_iterations set to a default value: " + str(ITERATIONS)) 

start = time.time()

# Initialize samples
# [x1, y1, z1] 
# [x2, y2, z2] 
# [x3, y3, z3] 
#  ... .. ...
# [xn, yn, zn] 
#
samples = tf.Variable(tf.random_uniform([NUM_SAMPLES,DIM]))

# Initialize cluster assignments
# [c1]
# [c2]
# [c3]
#  ..
# [cn]
cluster_assignments = tf.Variable(tf.zeros([NUM_SAMPLES], dtype=tf.int64))

# Initialize centroid matrix (Assume 4)
# [kx1, ky1, kz1]
# [kx2, ky2, kz2]
# [kx3, ky3, kz3]
# [kx4, ky4, kz4]
# 
centroids = tf.Variable(tf.random_uniform([NUM_CLUSTERS, DIM]))

# Tile sample
# [x1, y1, z1]
# [x1, y1, z1] 
# [x1, y1, z1] 
# [x1, y1, z1] 
# [x2, y2, z2] 
# [x2, y2, z2] 
# [x2, y2, z2] 
# [x2, y2, z2] 
#  ... .. ...
# [xn, yn, zn] 
#
tile_samples = tf.tile(samples, [1, NUM_CLUSTERS])
tile_samples = tf.reshape(tile_samples, [NUM_SAMPLES * NUM_CLUSTERS, DIM])

# Tile centroids
# [kx1, ky1, kz1]
# [kx2, ky2, kz2]
# [kx3, ky3, kz3]
# [kx4, ky4, kz4]
# [kx1, ky1, kz1]
# [kx2, ky2, kz2]
# [kx3, ky3, kz3]
# [kx4, ky4, kz4]
#  ...  ...  ...
tile_centroids = tf.tile(centroids, [NUM_SAMPLES, 1])
tile_centroids = tf.reshape(tile_centroids, [NUM_SAMPLES * NUM_CLUSTERS, DIM])

# Multiply element-wise and sum the rows
tile_sum = tf.reduce_sum(tf.square(tile_samples - tile_centroids), 1)

# Reshape to align each cluster calcualtion
reshape_sum = tf.reshape(tile_sum, [NUM_SAMPLES, NUM_CLUSTERS])

# Determine new cluster assignments
cluster_assignments = tf.argmin(reshape_sum, axis=1)

# Sum datapoints based on cluster assignments
# Then, divide by count to calculate new centroids
# Source: https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/segmentation
new_centroids = tf.unsorted_segment_sum(samples, cluster_assignments, NUM_CLUSTERS)
count_cluster_assignments = tf.unsorted_segment_sum(tf.ones([NUM_SAMPLES, DIM]), cluster_assignments, NUM_CLUSTERS)
new_centroids = new_centroids / count_cluster_assignments

# Assign the new centroids
assign_new_centroids = tf.assign(centroids, new_centroids)

# Run the clustring algorithm 
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(0, ITERATIONS):
    centroids = sess.run(assign_new_centroids)

[assignments, centroids] = sess.run([cluster_assignments, new_centroids])

end = time.time()
print("Time to calculate clusters: " + str(end-start) + " Seconds")

# Show a graph if the number of clusters and number of samples is reasonably small
if NUM_CLUSTERS <= 6 & NUM_SAMPLES <= 200:
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow']

    plot_samples = sess.run(samples)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, NUM_SAMPLES):
        ax.scatter(plot_samples[i,0], plot_samples[i,1], plot_samples[i, 2], color=colors[assignments[i]])

    for centroid in range(0, NUM_CLUSTERS):
        ax.scatter(centroids[centroid,0], centroids[centroid,1], centroids[centroid,2], marker='x', color=colors[centroid])

    plt.show()
