import numpy as np  # importing the numpy library
from mnist import MNIST  # importing the mnist library
import matplotlib.pyplot as plt  # import libraries to display the cube
from PIL import Image

# definition for importing drawings from photoshop:
def photoshopimage(filepath): # Achtung! Beim Filepath \\ einfügen!
    pic = Image.open(filepath) # open the image
    pix = np.array(pic) # convert into a np. array
    pix = pix[:,:,0] < 32 # choose one layer of values and only keep the pixels with values lower than 32 (the black values)
    return(1*pix) # 1 * pix to convert the true and false values into 1 and 0


def mnistimage(num): # definition to import arrays of mnist
    img = mnist.train_set.images[num, :].reshape(28,28)  # Take image num+1 of the training set and reshape its array to a 28 * 28 grid
    a = mnist.train_set.labels[num, :]  # assign the corresponding label to a
    print(np.where(a == 1)[0][0])  # printing the position of label a which corresponds to the number in the img1
    # [0][0] converts it to an integer
    return(img)

def perlenoptimieren(cube): # definition to minimize the number of points plotted
    sz = cube.shape
    for z in range(sz[2]): # für alle Ebenen in die z Richtung
        for x in range(sz[0]): # für alle Zeilen in die x Richtung.
            for y in range(sz[1]): # für alle Spalten in die y Richtung.
                if cube[x, y, z] == 1:
                    row = cube[x, :, z]
                    column = cube[:, y, z]
                    if sum(row) > 1 and sum(column) > 1: # Wenn in der gleichen Reihe und Spalte auch bereits eine 1 ist,
                        cube[x, y, z] = 0 # Dann wird die 1 zu einer 0 umgewandelt.
    return(cube)

# definition to make a construction plan for my model with pearls
def printBauplan(cubegraph):
    g = np.arange(cubegraph.shape[2])  # make an array the length of the z axes

    for x in range(cubegraph.shape[0]):  # check all combinations of x and y coordinates in cubegraph
        for y in range(cubegraph.shape[1]):
            v = g[cubegraph[x, y,:]]  # g assigns a number from the array to each true value on the z axes (v) at point x,y
            if len(v) > 0:  # if there are true values, it prints the z coordinate of these values. (seen from bottom to top)
                ebene = (x, y, v)
                print(ebene)

def Modellbau(cubegraph): # prints graphically construction plan for the pearlmodels
    for x in range(cubegraph.shape[0]-1,-1,-1): # for each layer when the cube is flipped
        Ebene = ""
        for y in range(cubegraph.shape[1]):  # check all combinations of y and z coordinates in cubegraph
            for z in range(cubegraph.shape[2]):
                if cubegraph[x,y,z]==1: # if there is a value, add X to the string. These are where the pearls will be.
                    Ebene += " X"
                else:
                    Ebene += " ."
            Ebene += "\n"
        print(Ebene)


def optimizeAllSides(cube): # optimize the number of points starting from the 4 different sides of the cube
    cube1 = perlenoptimieren(cube)
    cube2 = np.rot90(cube, 1)
    cube2 = perlenoptimieren(cube2)
    cube3 = np.rot90(cube, 2)
    cube3 = perlenoptimieren(cube3)
    cube4 = np.rot90(cube, 3)
    cube4 = perlenoptimieren(cube4)

    sz = cube.shape
    for z in range(sz[2]): # für jede Ebene die Anzahl Punkte zusammenzählen
        a = np.sum(cube1[:, :, z])
        b = np.sum(cube2[:, :, z])
        c = np.sum(cube3[:, :, z])
        d = np.sum(cube4[:, :, z])
        min_index = np.argmin([a, b, c, d]) # Die Ebene herausgeben, die die minimalste Anzahl Punkte hat.
    return(cube1,cube2,cube3,cube4)


def calculatemin (cube): # calculating the minimal number of points necessary to display this model
    print("Die Minimums der Ebenen sind:")
    sz = cube.shape
    summe = 0
    for z in range(sz[2]): # für jede Ebene
        a = np.sum(cube[:,:,z],0) # Die Summe der Punkte in allen Reihen in x - Richtung zusammenzählen.
        b = np.sum(cube[:,:,z],1) # Die Summe der Punkte in allen Spalten in y - Richtung zusammenzählen.

        c = np.size(np.where(a > 0)) # Die Anzahl Reihen zusammenzählen, in denen a > 0
        d = np.size(np.where(b > 0)) # Die Anzahl Spalten zusammenzählen in denen b > 0

        if c > d: # Je nachdem ob c oder d grösser ist, entspricht dies der minimaler möglicher Anzahl Punkte.
            print( "Ebene:", z,":", c)
            summe= summe+c

        else:
            print("Ebene",z, ":", d)
            summe = summe + d
    print("Das Optimum besteht aus",summe,"Punkten.")
    return (a, b, c, d,summe)



def CalcMinCube (cube): # calculating the minimal number of points necessary to display this model
    sz = cube.shape
    optimum = np.zeros(sz) # creating a copy of cube, consisting only of 0's
    summeebene = 0
    for z in range(sz[2]): # für jede Ebene
        a = np.sum(cube[:,:,z],0) # Die Summe der Punkte in allen Reihen in x - Richtung zusammenzählen.
        b = np.sum(cube[:,:,z],1) # Die Summe der Punkte in allen Spalten in y - Richtung zusammenzählen.

        c = np.size(np.where(a > 0)) # Die Anzahl Reihen zusammenzählen, in denen a > 0
        d = np.size(np.where(b > 0)) # Die Anzahl Spalten zusammenzählen in denen b > 0

        if d >= c: # Je nachdem ob c oder d grösser ist, entspricht dies der minimalen möglicher Anzahl Punkte.
            aa = np.zeros(a.shape)
            for x in range(sz[0]):

                if b[x] > 0: # a an der Stelle x
                    pointSetFlag = False

                    for y in range(sz[1]):
                        if (cube[x,y,z]>0) and (aa[y] == 0):
                            optimum[x, y, z] = 1
                            aa[y] = +1
                            pointSetFlag = True
                            break
                    if pointSetFlag == False:
                        for y in range(sz[1]):
                            if (cube[x, y, z] > 0):
                                optimum[x, y, z] = 1
                                aa[y] = +1
                                pointSetFlag = True
                                break

                        print()

        else: # man könnte den Würfel hier auch 90 Grad rotieren und das Selbe wie oben machen und das Resultat anschliessend zurück rotieren.
            bb = np.zeros(b.shape)
            for y in range(sz[1]):
                if a[y]>0:
                    pointSetFlag = False

                    for x in range(sz[0]):

                        if (cube[x, y, z] > 0) and (bb[x] == 0):
                            optimum[x, y, z] = 1
                            bb[x] = +1
                            pointSetFlag = True
                            break
                    if pointSetFlag == False:
                        for x in range(sz[0]):
                            if (cube[x, y, z] > 0):
                                optimum[x, y, z] = 1
                                bb[x] = +1
                                pointSetFlag = True
                                break


    return(optimum)


def randomCalcMinCube (cube): # Randomizing the pearl distribution in CalcMinCube
    sz = cube.shape
    optimum = np.zeros(sz) # creating a copy of cube, consisting only of 0's
    for z in range(sz[2]): # für jede Ebene
        a = np.sum(cube[:,:,z],0) # Die Summe der Punkte in allen Reihen in x - Richtung zusammenzählen.
        b = np.sum(cube[:,:,z],1) # Die Summe der Punkte in allen Spalten in y - Richtung zusammenzählen.

        c = np.size(np.where(a > 0)) # Die Anzahl Reihen zusammenzählen, in denen a > 0
        d = np.size(np.where(b > 0)) # Die Anzahl Spalten zusammenzählen in denen b > 0

        if d >= c: # Je nachdem ob c oder d grösser ist, entspricht dies der minimalen möglicher Anzahl Punkte.
            aa = np.zeros(a.shape)
            for x in range(sz[0]):
                if b[x] > 0: # b an der Stelle x
                    pointlist = np.array([], dtype = 'int') # Eine leere Liste erzeugen
                    for y in range(sz[1]): # Alle Punkte in dieser Spalte/Zeile durchgehen
                        if (cube[x,y,z]>0): # Wenn es im Originalcube dort einen Punkt hat, so wird diese y-Stelle in die Liste aufgenommen.
                            pointlist = np.append(pointlist,y)

                    np.random.shuffle(pointlist) # Die Elemente der Liste mischen.
                    pointSetFlag = False
                    for y in list(pointlist): # Jedes Element der Liste durchgehen.
                        if aa[y] == 0: # Wenn es in dieser Zeile noch keinen anderen Punkt hat,
                            optimum[x, y, z] = 1 #... so wird hier der Punkt gesetzt.
                            aa[y] = +1 # Damit in Zukunft keine Punkte mehr in diese Zeile gesetzt werden, wird an dieser Stelle im aa eine 1 gesetzt.
                            pointSetFlag = True
                            break
                    if pointSetFlag == False: # Wenn kein Punkt gesetzt wurde, so heisst das, dass es in jeder Zeile bereits einen Punkt gab.
                        y = pointlist[0] # Dann ist es eigentlich egal, wo der Punkt gesetzt wird. Nehmen wir einfach [0]
                        optimum[x,y,z]=1
                        #aa[y]= +1


        else: # Der selbe Prozess wie oben, nur dass wir von einer anderen Seite her durch die Punkte gehen.
            bb = np.zeros(b.shape)
            for y in range(sz[1]):
                if a[y]>0:
                    pointlist = np.array([], dtype='int')
                    for x in range(sz[0]):
                        if (cube[x, y, z] > 0):
                            pointlist = np.append(pointlist,x)

                    np.random.shuffle(pointlist)
                    pointSetFlag = False
                    for x in list(pointlist):
                        if bb[x] == 0:
                            optimum[x, y, z] = 1
                            bb[x] = +1
                            pointSetFlag = True
                            break
                    if pointSetFlag == False:
                            x = pointlist[0]
                            optimum[x, y, z] = 1
                            #bb[x] = +1



    return(optimum)





if __name__ == "__main__": # Dieser untere Teil wird nur ausgeführt, wenn ich testing1 runnen lasse. Aber nicht, wenn ich es als Modul importiere.
     # training set with letters
    mnist = MNIST()
    #mnist.train_set.images  #(60000, 784) #(60000 images, 784 values for the 28*28 grid; arranged as array)
    #mnist.train_set.labels  (60000, 10) #(60000 images, 10 different possible labels -one hot encoded (for deep learning))



    img1 = photoshopimage("C:\\Users\\Celine\\Documents\\Kanti Jahr 3\\Maturaarbeit\\png files\\C.png")
    img2 = photoshopimage ("C:\\Users\\Celine\\Documents\\Kanti Jahr 3\\Maturaarbeit\\png files\\A.png")

    #img1 = mnistimage(7)
    #img2 = mnistimage(9)



    img1 = np.rot90(img1, -1, (0, 1))  # rotate the image
    vol2 = np.tile(img1, (img1.shape[0], 1, 1))  # img1 is stacked on top of each other the number of times equal the length of the img1


    img2 = np.rot90(img2, -1, (0, 1))  # rotate image
    vol1 = np.tile(img2, (img2.shape[0], 1, 1))  # img2 ist stacked on top of each other the number of times equal the length of the img2.
    vol1 = np.rot90(vol1, 1)  # rotate vol2 to show the number from another side as vol1
    # print(np.where(b == 1)[0][0])  # printing the position of label b which corresponds to the number of img2

    a = np.sum(img1, 0)  # Die Summe der Punkte in allen Spalten in y - Richtung zusammenzählen.
    b = np.sum(img2, 0)  # Die Summe der Punkte in allen Spalten in y - Richtung zusammenzählen.
    c = np.sign(a)  # Alle Werte > 0 werden in eine 1 umgewandelt, die anderen in eine 0
    d = np.sign(b)

    if not np.all(c == d):  # Kleine Kontrollfunktion falls eine Ansicht, leere Ebenen hat und somit einen Teil der anderen Ansicht auslöscht.
         print("Die Ansicht ist unvollständig.")


    cube = vol1 * vol2  # multiplying the two cubes of the two images
    cubegraph = cube > 0  # print only the points where there is a number > 1 (because 0 = no color, 1= black )
    cube[cubegraph] = 1.  # converting all values that are true in cubegraph to a 1.


    calculatemin(cube)


    #cube1, cube2, cube3, cube4 = optimizeAllSides(cube) # Die 4 verschiedenen Cubes herausgeben.


    #cube = perlenoptimieren(cube)
    optimum = randomCalcMinCube(cube)
    cubegraph = optimum > 0  # print only the points where there is a number > 0 (because 0 = no color, 1= black )

    anzahlperlen = np.size(np.where(cubegraph == True))/3  # count the printed points. /3 because np.where gives all three coordinates for every point.
    print("Die Anzahl Perlen ist:", anzahlperlen)
    Modellbau(cubegraph)
    #printBauplan(cubegraph)

    # Create a meshgrid for x, y, z coordinates
    x, y, z = np.meshgrid(np.arange(cubegraph.shape[0]),# make a meshgrid 28*28 (the shape of cubegraph of the x axes(x=0))
                          np.arange(cubegraph.shape[1]),
                          np.arange(cubegraph.shape[2]))

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # put the plot in the field 111 (because I have only 1 plot)

    # Plot the points
    ax.scatter(x[cubegraph], y[cubegraph], z[cubegraph])  # plot the points where cubegraph is true

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


    # convert numpy array into STL file
    from skimage import measure  # skimage is for image processing, measure can measure image properties
    from stl import mesh  # import numpy-stl library

    # Generate vertices and faces using marching_cubes
    verts, faces, normals, values = measure.marching_cubes(cubegraph, spacing=(1, 1, 1))

    # Create the mesh
    surf = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            surf.vectors[i][j] = verts[f[j], :]

    # Save as STL file
    surf.save('try.stl')