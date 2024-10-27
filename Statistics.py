##################################################################
# Programmteil "Statistics"
# Anhand aller Buchstabenkombinationen der Grossbuchstaben wird
# statistisch ausgewertet, wie effektiv die einzelnen Optimierungsalgorithmen durchschnittlich sind

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
#   Input:  text - eine Stellenangabe aus einem Buchstabenstring
#   Output: Bild des Buchstabens als 28 * 28 NumPy-Array
#   Beschreibung: Generiert Bilder mit einem spezifischen Text als 28 * 28 NumPy-Array
def LetterArray(text):

    img = Image.new(mode="RGB", size=(64, 64), color=(255, 255, 255))  # Weisses Bild erstellen
    draw = ImageDraw.Draw(img)  # Weisses Bild darstellen
    font = ImageFont.truetype('arial.ttf', 64)  # Font und Schriftgrösse einstellen

    draw.text((0, 0), text, fill="black", font=font)  # Text zeichnen
    bbox = font.getbbox(text)  # Zeichen-Box des Textes berechnen
    img1 = img.crop(bbox)  # Bild auf Grösse der Zeichen-Box schneiden

    img2 = img1.resize((28, 28))  # Zu einem 28*28 Array in der Grösse ändern
    pix = np.array(img2)  # img2 in NumPy-Array umwandeln
    img2 = pix[:, :, 0]  # Aus dem RGB Bild eine Ebene aussuchen

    if picture:  # Bild darstellen, wenn picture == True ist
        img2.show()

    return (img2)  # Bild des Buchstabens zurückgeben


#################################################################
# Funktion AveragePoints()
#   Input:  Keiner
#   Output: Ein data frame mit den Anzahl Punkten für jedes Paar für alle Optimierungsalgorithmen
#   Beschreibung:  Speichern der Anzahl Punkte für alle möglichen Paare von Grossbuchstaben
#   beim Durchgehen durch jeden Optimierungsmechanismus

def AveragePoints():

    columns = ["Letters","None","Front", "Back","Left", "Right", "Optimum"]  # Spalten des data frames erstellen
    df = pd.DataFrame(columns=columns)  # Data frame erstellen

    for x in range(26):
        for y in range(26):  # Für alle möglichen Paare von zwei Bildern (In diesem Fall von Grossbuchstaben)
            a = capital_letters[x]+capital_letters[y]  # Die beiden Buchstaben in der ersten Spalte "Letters" speichern
            print(a)  # Buchstabenpaar ausgeben

            img1 = LetterArray(capital_letters[x])  # Mit Funktion LetterArray aus dem Buchstaben ein Bild erstellen
            img2 = LetterArray(capital_letters[y])

            vol1 = np.tile(img1, (28, 1, 1))  # img1 28 Mal aufeinanderstappeln, um einen Würfel zu bilden
            vol2 = np.tile(img2, (28, 1, 1))  # img2 28 Mal aufeinanderstappeln, um einen Würfel zu bilden
            vol2 = np.rot90(vol2, 1)  # vol2 90 Grad rotieren, um Bild von einer anderen Seite als vol1 darzustellen

            cube = vol1 * vol2  # Würfel der beiden Bilder miteinander multiplizieren
            cubegraph = cube > 0  # Nur Punkte darstellen, in denen eine Zahl > 1 ist (denn 0 = weiss , 1 = schwarz)
            cube[cubegraph] = 1.  # Werte, die true sind in cubegraph, in eine 1 umwandeln

            # Anzahl Punkte ohne Optimierungsverfahren zählen,
            # in Spalte "none" speichern
            b = np.size(np.where(cubegraph == 1))

            ####### Erstes Optimierungsverfahren:
            PerlenOptimieren(cube)
            c = np.size(np.where(cube == 1))  # Anzahl Punkte nach Verfahren zählen, in dritter Spalte "Front" speichern

            ####### Zweites Optimierungsverfahren: Würfel von allen Seiten durch Funktion PerlenOptimieren lassen,
            # um Anzahl Punkte zu vergleichen
            cube2 = np.rot90(cube, 1)  # Würfel ein Mal um 90 Grad im Gegenuhrzeigersinn drehen
            PerlenOptimieren(cube2)  # Um Funktion PerlenOptimieren von einer anderen Seite her zu starten
            f = np.size(np.where(cube2 == 1))  # Anzahl Punkte in der Spalte "Right" speichern

            cube3 = np.rot90(cube, 2)  # Würfel zwei Mal rotieren, um von hinten zu starten
            PerlenOptimieren(cube3)
            d = np.size(np.where(cube3 == 1))  # Anzahl Punkte in der Spalte "Back" speichern

            cube4 = np.rot90(cube, 3)  # Würfel drei Mal rotieren
            PerlenOptimieren(cube4)
            e = np.size(np.where(cube4 == 1))  # Anzahl Punkte in der Spalte "Left" speichern

            ###### Drittes Optimierungsverfahren:
            optimum = PlotMinCube(cube)
            g = np.size(np.where(optimum == 1))  # Anzahl Punkte in der Spalte "Optimum" speichern

            # Neue Spalte erstellen mit den Daten von diesem Buchstabenpaar
            new_row = {'Letters': a, 'None': b, 'Front': c, 'Back': e, 'Left': f, 'Right': d, 'Optimum': g}
            df = df._append(new_row, ignore_index=True)  # Diese neue Zeile dem data frame hinzufügen
    return (df)


capital_letters = string.ascii_uppercase[:]  # Grossbuchstaben in der Variabel capital_letters speichern
df = AveragePoints()

if excel == True:
    df.to_excel("Statistic.xlsx")  # Data frame als Excel Tabelle abspeichern