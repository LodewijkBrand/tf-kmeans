import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DIM = 2
NUM_SAMPLES = 100
NUM_CLUSTERS = 3

def create_samples(num_points, dimension):
    return np.random.rand(num_points, dimension)

def create_centroids(num_clusters, dimension):
    return np.random.rand(num_clusters, dimension)

def getLabels(samples, centroids, sess):
    labels = np.zeros((NUM_SAMPLES, NUM_CLUSTERS)) 

    for i in range(0, NUM_SAMPLES):
        sample = samples[i,:]
        centroid_num = 0
        min_distance = 100
        for s in range(0, NUM_CLUSTERS):
            centroid = centroids[s,:]
            distance = sess.run(tf.norm(centroid - sample))
            if (distance < min_distance):
                min_distance = distance
                centroid_num = s
        labels[i,centroid_num] = 1.0

    return labels

def getCentroids(samples, labels, sess):
    centroids = sess.run(tf.matmul(tf.transpose(labels), samples))
    centroids = sess.run(tf.transpose(tf.divide(tf.transpose(centroids), tf.reduce_sum(labels, 0)))) 
    return centroids

samples = create_samples(NUM_SAMPLES, DIM)
centroids = create_centroids(NUM_CLUSTERS, DIM)

with tf.Session() as sess:
    for i in range(0, 5):
        labels = getLabels(samples, centroids, sess)
        centroids = getCentroids(samples, labels, sess)
        print "epoch: " + str(i)

colors = ['red', 'green', 'blue', 'purple']

for sample in range(0, NUM_SAMPLES):
    sample_color = 0
    for cluster in range(0, NUM_CLUSTERS):
        if labels[sample,cluster] == 1:
            sample_color = cluster

    plt.scatter(samples[sample,0], samples[sample,1], color=colors[sample_color])

for centroid in range(0, NUM_CLUSTERS):
    plt.scatter(centroids[centroid,0], centroids[centroid,1], marker='x', color=colors[centroid])

plt.ylabel('some numbers')
plt.show()
