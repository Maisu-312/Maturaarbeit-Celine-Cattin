##################################################################
# Programmteil "Statistics"
# Anhand aller Buchstabenkombinationen der Grossbuchstaben wird
# statistisch ausgewertet,(komma?) wie effektiv die einzelnen Optimierungsalgorithmen durchschnittlich sind.

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string
import pandas as pd
from MaturaarbeitCélineCattin import PerlenOptimieren
from MaturaarbeitCélineCattin import PlotMinCube

### Start Konfiguration

# Die Bilder darstellen:
picture = False

# Eine Excel Tabelle mit den Daten ausgeben:
excel = True

# Ende Konfiguration


#################################################################
# Funktion LetterArray(text)
#   Input:  text - eine Stellenangabe aus einem Buchstabenstring oder einfach ein Buchstabe ?
#   Output: Bild des Buchstabens als 28 * 28 NumPy-Array
#   Beschreibung: Generiert Bilder mit einem spezifischen Text als 28 * 28 NumPy-Array.
def LetterArray(text):

    img = Image.new(mode="RGB", size=(64, 64), color=(255, 255, 255))  # Ein weisses Bild erstellen - muss nach jedem Satz ein Punkt hin?
    draw = ImageDraw.Draw(img)  # Das weisse Bild darstellen
    font = ImageFont.truetype('arial.ttf', 64)  # Den Font und die Schriftgrösse einstellen

    draw.text((0, 0), text, fill="black", font=font)  # Den Text zeichnen
    bbox = font.getbbox(text)  # Die Zeichen-Box des Textes berechnen
    img1 = img.crop(bbox)  # Das Bild auf die Grösse der Zeichen-Box schneiden

    img2 = img1.resize((28, 28))  # Zu einem 28*28 Array in der Grösse ändern.

    if picture:  # Die Bilder darstellen ( ist das gut formuliert so?) , wenn picture = True ist. (= oder == ?)
        img2.show()

    return (img2)  # Das Bild des Buchstabens zurückgeben.


#################################################################
# Funktion AveragePoints()
#   Input:  Keiner
#   Output: Ein data frame mit den Anzahl Punkten für jedes Paar für alle Optimierungsalgorithmen
#   Beschreibung:  Speichern der Anzahl Punkte für alle möglichen Paare von Grossbuchstaben beim Durchgehen durch jeden Optimierungsmechanismus.

def AveragePoints():

    columns = ["Letters","None","Front", "Back","Left", "Right", "Optimum"]  # Die Spalten des data frames erstellen
    df = pd.DataFrame(columns=columns)  # Das data frame erstellen

    for x in range(26):
        for y in range(26):  # Für alle möglichen Paare von zwei Bildern ( In diesem Fall von Grossbuchstaben)
            a = capital_letters[x]+capital_letters[y]  # Die beiden Buchstaben in die erste Spalte "Letters" speichern
            print(a)  # Das Buchstabenpaar ausgeben

            img1 = LetterArray(capital_letters[x]) #  Mit Hilfe der Funktion letterarray aus dem Buchstaben ein Bild erstellen.
            img2 = LetterArray(capital_letters[y])

            pix = np.array(img1)  # img1 in ein NumPy-Array umwandeln (weshalb ist das nötig?)
            img1 = pix[:, :, 0]  # Aus dem RGB Bild eine Ebene aussuchen

            pix2 = np.array(img2)  # img2 in ein NumPy-Array umwandeln
            img2 = pix2[:, :, 0]  # Aus dem RGB Bild eine Ebene aussuchen

            vol1 = np.tile(img1, (28, 1, 1))  # img1 28 Mal aufeinanderstappeln, um einen Würfel zu bilden
            vol2 = np.tile(img2, (28, 1, 1))  # img2 28 Mal aufeinanderstappeln, um einen Würfel zu bilden
            vol2 = np.rot90(vol2, 1)  # vol2 um 90 Grad rotieren, um das Bild von einer anderen Seite als vol1 darzustellen

            cube = vol1 * vol2  # Die Würfel der beiden Bilder miteinander multiplizieren
            cubegraph = cube > 0  # Nur Punkte darstellen, in denen eine Zahl > 1 ist. ( denn 0 = weiss , 1 = schwarz)
            cube[cubegraph] = 1.  # Alle Werte, die true sind in cubegraph in eine 1 umwandeln.

            b = np.size(np.where(cubegraph == 1))  # Die Anzahl Punkte ohne Optimierungsverfahren zählen und in der zweiten Spalte "none" speichern.

            ####### Den Würfel durch das erste Optimierungsverfahren lassen
            PerlenOptimieren(cube)
            c = np.size(np.where(cube == 1))  # Die Anzahl Punkte nach dem Verfahren zählen und in der dritten Spalte "Front" speichern.

            ####### Für das zweite Optimierungsverfahren wird der Würfel von allen Seiten her durch die Funktion perlenoptimieren gelassen, um die Anzahl Punkte zu vergleichen.
            cube2 = np.rot90(cube, 1)  # Den Würfel ein Mal um 90 Grad im Gegenuhrzeigersinn drehen rotieren,
            PerlenOptimieren(cube2)  # um die Funktion perlenoptimieren von einer anderen Seite her zu starten.
            f = np.size(np.where(cube2 == 1))  # Die Anzahl Punkte in der Spalte "Right" speichern

            cube3 = np.rot90(cube, 2)  # Würfel zwei Mal rotieren, um von hinten zu starten
            PerlenOptimieren(cube3)
            d = np.size(np.where(cube3 == 1))  # Die Anzahl Punkte in der Spalte "Back" speichern

            cube4 = np.rot90(cube, 3)  # Den Würfel drei Mal rotieren
            PerlenOptimieren(cube4)
            e = np.size(np.where(cube4 == 1))  # Die Anzahl Punkte in der Spalte "Left" speichern

            ###### Würfel durch das dritte Optimierungsverfahren lassen
            optimum = PlotMinCube(cube)
            g = np.size(np.where(optimum == 1))  # Die Anzahl Punkte in der Spalte "Optimum" speichern

            new_row = {'Letters': a, 'None': b, 'Front': c, 'Back': e, 'Left': f, 'Right': d, 'Optimum': g}  # Eine neue Spalte erstellen mit den Daten von diesem Buchstabenpaar.
            df = df._append(new_row, ignore_index=True)  # Diese neue Zeile zum data frame hinzufügen
    return (df)


capital_letters = string.ascii_uppercase[:]  # Die Grossbuchstaben in der Variabel capital_letters speichern
df = AveragePoints()  # Die Funktion averagepoints aufrufen

if excel == True:
    df.to_excel("Statistic.xlsx")  # Das data frame als Excel Tabelle abspeichern