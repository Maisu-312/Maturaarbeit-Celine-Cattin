# def letterarray generates images (size 28*28) depicting a certain text.
# def averagepoints generates all possible pairs of the uppercase letters, turns them into images using def letterarray
#   and then applies all optimization functions storing the data in a data frame and excel sheet.
# The aim is to have an overview of how effective these optimization functions are in average.

import numpy as np  # importing the numpy library
from PIL import Image, ImageDraw, ImageFont
import string
import pandas as pd
from testing1 import perlenoptimieren
from testing1 import CalcMinCube

def letterarray(text, debug = False): # A function to generate images (size 28*28) depicting a certain text

    img = Image.new(mode="RGB", size=(64, 64), color= (255,255,255)) # Creating a white image

    draw = ImageDraw.Draw(img) # Depicting the image
    font = ImageFont.truetype('arial.ttf', 64) # Setting the font and size


    draw.text((0,0), text, fill = "black", font = font) # Drawing the text
    bbox = font.getbbox(text) # Calculating the bounding box of the given text
    img1 = img.crop(bbox) # Cropping the image to the size of this bounding box

    img2 = img1.resize((28, 28))  # Resizing the image to the size of 28 * 28

    if debug: # Only showing the image if debug = True.
        img2.show()

    return (img2)



def averagepoints(): # Definition to collect the amoung of points of each possible pairing of the alphabet when going through the optimization functions

    columns = ["Letters","None","Front", "Back","Left", "Right", "Optimum"] # Creating the columns for the data frame
    df = pd.DataFrame(columns=columns) # Creating the data frame

    for x in range(26):
        for y in range (26): # For all possible pairings of two images (in this case the uppercase letters):
            a = capital_letters[x]+capital_letters[y] # Printing the two letters into the first column
            print(a)

            img1 = letterarray(capital_letters[x])
            img2 = letterarray(capital_letters[y])

            pix = np.array(img1)
            img1 = pix[:, :, 0]   # Choosing one layer of the RGB picture

            pix2 = np.array(img2)
            img2 = pix2[:, :,0]

            vol1 = np.tile(img1, (28, 1, 1))  # img1 is stacked on top of each other 28 times
            vol2 = np.tile(img2, (28, 1, 1))  # img2 ist stacked on top of each other 28 times
            vol2 = np.rot90(vol2, 1)  # Rotate vol2 to show the text from another side as vol1

            cube = vol1 * vol2  # Multiplying the two cubes of the two images
            cubegraph = cube > 0  # Print only the points where there is a number > 1 (because 0 = no color, 1= black )
            cube[cubegraph] = 1.  # Converting all values that are true in cubegraph to a 1.

            b = np.size(np.where(cubegraph == 1))  # Count the printed points with no optimization function

            perlenoptimieren(cube)
            c = np.size(np.where(cube == 1))  # Count the printed points after the perlenoptimieren function

            cube2 = np.rot90(cube, 1) # Rotate the cube to start the perlenoptimieren function from another side
            perlenoptimieren(cube2)
            d = np.size(np.where(cube2 == 1 ))

            cube3 = np.rot90(cube, 2)
            perlenoptimieren(cube3)
            e = np.size(np.where(cube3 == 1))

            cube4 = np.rot90(cube, 3)
            perlenoptimieren(cube4)
            f = np.size(np.where(cube4 == 1))

            optimum = CalcMinCube(cube)
            g = np.size(np.where(optimum == 1))

            new_row = {'Letters': a , 'None': b , 'Front': c , 'Back': e, 'Left' : f, 'Right': d, 'Optimum': g} # Create a new row with the data from this image pair
            df = df._append(new_row,ignore_index=True) # Add this row to the data frame
    return(df)

capital_letters = string.ascii_uppercase[:]  # Save the uppercase capital letters in variable capital_letters


df = averagepoints()
df.to_excel("Statistic.xlsx") # Saving the data frame as an Excel sheet