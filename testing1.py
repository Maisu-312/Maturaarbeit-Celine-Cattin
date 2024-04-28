import numpy as np #importing the numpy library
from mnist import MNIST #importing the mnist library


# training set with letters
mnist = MNIST()
# mnist.train_set.images  (60000, 784) (60000 images, 784 values for the 28*28 grid; arranged as array)
# mnist.train_set.labels  (60000, 10) (60000 images, 10 different possible labels -one hot encoded (for deep learning))

num1 = 4
img1 = mnist.train_set.images[num1,:].reshape(28,28) # Take image num1+1 of the training set and reshape its array to a 28 * 28 grid
img1 = np.rot90 (img1,-1,(0,1)) # rotate the image
a = mnist.train_set.labels[num1,:] # assign the corresponding label to a
vol1 = np.tile(img1,(28,1,1)) # img1 is stacked on top of each other 28 times
print(np.where(a == 1)[0][0]) # printing the position of label a which corresponds to the number in the img1
# [0][0] converts it to an integer

num2 = 9
img2 = mnist.train_set.images[num2,:].reshape(28,28) # Take image num2+1 of training set and reshape its array to a 28*28 grid
img2 = np.rot90 (img2,-1,(0,1)) # rotate image
b = (mnist.train_set.labels[num2,:]) # assign the corresponding label to b
vol2 = np.tile(img2,(28,1,1)) # img2 ist stacked on top of each other 28 times
vol2 = np.rot90(vol2,1) # rotate vol2 to show the number from another side as vol1
print(np.where(b ==1)[0][0]) # printing the position of label b which corresponds to the number of img2

cube = vol1 * vol2 # multiplying the two cubes of the two images
cubegraph = cube > 0 # print only the points where there is a number > 1 (because 0 = no color, 1= black )
cube[cubegraph] = 1. # converting all values that are true in cubegraph to a 1.

print(cubegraph)

anzahlperlen=np.size(np.where(cubegraph==True)) # count the printed points
print(anzahlperlen)

import matplotlib.pyplot as plt # import libraries to display the cube


# Create a meshgrid for x, y, z coordinates
x, y, z = np.meshgrid(np.arange(cubegraph.shape[0]), # make a meshgrid 28*28 (the shape of cubegraph of the x axes(x=0))
                      np.arange(cubegraph.shape[1]),
                      np.arange(cubegraph.shape[2]))

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # put the plot in the field 111 (because I have only 1 plot)

# Plot the points
ax.scatter(x[cubegraph], y[cubegraph], z[cubegraph]) # plot the points where cubegraph is true


# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()

# definition to make a construction plan for my model with pearls
def printBauplan(cubegraph):
    g = np.arange(cubegraph.shape[2]) # make an array the length of the z axes

    for x in range(cubegraph.shape[0]): # check all combinations of x and y coordinates in cubegraph
        for y in range(cubegraph.shape[1]):
            v = g[cubegraph[x,y,:]] # g assigns a number from the array to each true value on the z axes (v) at point x,y
            if len(v) > 0: # if there are true values, it prints them.
                print(x,y,v)

printBauplan(cubegraph)