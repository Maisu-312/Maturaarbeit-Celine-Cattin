import numpy as np
from mnist import MNIST


mnist = MNIST()

# Train set is lazily loaded into memory and cached afterward
mnist.train_set.images  # (60000, 784)
mnist.train_set.labels  # (60000, 10)

# Test set is lazily loaded into memory and cached afterward
mnist.test_set.images   # (10000, 784)
mnist.test_set.labels   # (10000, 10)

# Yield minibatches from the shuffled train set
#for images, labels in mnist.train_set.minibatches(batch_size=256):
#    pass

img = mnist.train_set.images[4,:].reshape(28,28)
img = np.rot90 (img,-1,(0,1))
a = mnist.train_set.labels[0,:]
vol1 = np.tile(img,(28,1,1))
print(np.where(a ==1)[0][0])

img1 = mnist.train_set.images[9,:].reshape(28,28)
img1 = np.rot90 (img1,-1,(0,1))
b = (mnist.train_set.labels[2,:])
vol2 = np.tile(img1,(28,1,1))
vol2 = np.rot90(vol2,1)
print(np.where(b ==1)[0][0])

cube = vol1 * vol2
cubegraph = cube > 0
print(cube)

anzahlperlen=np.size(np.where(cubegraph==True))
print(anzahlperlen)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate some random 3D numpy array
array_3d = cubegraph

# Create a meshgrid for x, y, z coordinates
x, y, z = np.meshgrid(np.arange(array_3d.shape[0]),
                      np.arange(array_3d.shape[1]),
                      np.arange(array_3d.shape[2]))

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x[cubegraph], y[cubegraph], z[cubegraph], c=array_3d[cubegraph].flatten(), cmap='viridis')

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
