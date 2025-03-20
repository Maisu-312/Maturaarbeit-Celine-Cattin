#################################################################
# Maturaarbeit Céline Amélie Cattin
# 4. November. 2024
# Version 1.0

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from PIL import Image
import math
from skimage import measure
from stl import mesh
from skimage.transform import resize
from skimage.morphology import skeletonize


### Start Konfiguration

# Bilder Auswahl: Format: png, Zahl bedeutet Index in MNIST, String = Filename von zu ladendem Bild,
# Wenn man nur zwei Bilder möchte, beim Dritten "None" schreiben
# Achtung: Beim Filename muss der Pfad mit \\ untertrennt sein!
# Siehe Beispiel: "C:\\Users\\Celine\\Documents\\Kanti Jahr 3\\Maturaarbeit\\png files\\C.png"
imageA = "C:\\Users\\Celine\\Documents\\Kanti Jahr 3\\Maturaarbeit\\png files\\A.png"
imageB = "C:\\Users\\Celine\\Documents\\Kanti Jahr 3\\Maturaarbeit\\png files\\B.png"
imageC = "C:\\Users\\Celine\\Documents\\Kanti Jahr 3\\Maturaarbeit\\png files\\A.png"

# Grösse der Bilder (in Pixel): (nur quadratisches Format möglich)
size = (10,10)

# Threshold der Farbwerte: ( 0 <= threshold <= 1)
threshold_imageA = 0.5
threshold_imageB = 0.5
threshold_imageC = 0.5

# Angewendeter Optimierungsalgorithmus: 0 = nicht optimiert, 1 = Erstes Verfahren, 2 = Zweites Verfahren,
# 3 = Optimum Drittes Verfahren, 4 = Willkürliches Optimum
Optimierungsalgorithmus = 0

# Anzahl geplotteter Punkte ausgegeben:
AnzahlPunkte = True

# Drucken des Bauplanes für die Perlenmodelle:
PrintBauplan = False

# Anzahl Punkte im Optimum ausgeben:
MinimalePunkte = False

# Ausrechnen der Anzahl Möglichkeiten, um das Optimum darzustellen:
AnzahlMöglichkeiten = False

# Modell als STL-File ausgeben:
STL = False

# Modell als Drahtfigur visualisieren:
Drahtmodel = True

# Ende Konfiguration


#################################################################
# Funktion MnistImage(num)
#   Input: num - die Nummer des Bildes aus dem MNIST Datensatz
#   Output: Bild als NumPy-Array
#   Beschreibung: Lädt das num-te Bild aus dem MNIST Datensatz
def MnistImage(num):
    img = mnist.train_set.images[num, :].reshape(28, 28)  # Bild num+1 aus Trainingsset in ein 28*28 Array umformen
    a = mnist.train_set.labels[num, :]  # Zugehöriges Label von num wird a zugeordnet
    print(np.where(a == 1)[0][0])  # Position von Label a drucken, welche der Ziffer aus dem Bild entspricht
    return(img) # Zugehöriges MNIST Bild ausgeben


#################################################################
# Funktion LoadImage(filepath)
#   Input: filepath
#   Output: Bild als NumPy-Array
#   Beschreibung: Konvertiert importierte png Bilder zu einem NumPy-Array

def LoadImage(filepath,threshold):
    pic = Image.open(filepath)  # Bild öffnen
    pix = np.array(pic)  # In ein np.array konvertieren

    pix = resize(pix, size,  anti_aliasing=True )

    pix = pix[:, :, 0] < threshold # Eine Ebene von Werte aussuchen und Pixel behalten mit Werten < threshhold (schwarze Werte)
    return(1*pix) # 1 * pix um true und false Werte in 1 und 0 umzuwandeln


#################################################################
# Funktion PerlenOptimieren(cube)
#   Input: cube - dreidimensionales NumPy-Array der Projektionen durch den Würfel
#   Output: optimierter cube
#   Beschreibung: Minimiert die Anzahl geplotteter Punkte um durchschnittliche 92.14 %.
def PerlenOptimieren(cube):
    sz = cube.shape
    if imageC is not None:
        for z in range(sz[2]):  # Für alle Ebenen in die z Richtung
            for x in range(sz[0]):  # Für alle Zeilen in die x Richtung
                for y in range(sz[1]):  # Für alle Spalten in die y Richtung
                    if cube[x, y, z] == 1:  # Wenn an dieser Stelle ein Punkt (=1) gesetzt ist,
                        column = cube[x, :, z]
                        row = cube[:, y, z]
                        height = cube[x, y, :]
                        if sum(row) > 1 and sum(column) > 1 and sum(height) > 1:  # und in derselben Zeile und Spalte auch bereits ein Punkt ist,
                            cube[x, y, z] = 0  # dann wird die 1 zu einer 0 umgewandelt. - Punkt wird gelöscht
        return (cube)
    else:
        for z in range(sz[2]):  # Für alle Ebenen in die z Richtung
            for x in range(sz[0]):  # Für alle Zeilen in die x Richtung
                for y in range(sz[1]):  # Für alle Spalten in die y Richtung
                    if cube[x, y, z] == 1:  # Wenn an dieser Stelle ein Punkt (=1) gesetzt ist,
                        column = cube[x, :, z]
                        row = cube[:, y, z]
                        if sum(row) > 1 and sum(column) > 1: # und in derselben Zeile und Spalte auch bereits ein Punkt ist,
                            cube[x, y, z] = 0  # dann wird die 1 zu einer 0 umgewandelt. - Punkt wird gelöscht
        return(cube)


#################################################################
# Funktion OptimizeAllSides(cube)
#   Input: cube - dreidimensionales NumPy-Array der Projektionen durch den Würfel
#   Output: Vier cubes, optimiert startend von jeder Seite
#   Beschreibung: Optimiert die Anzahl Punkte mit der Funktion PerlenOptimieren startend von 4 verschiedenen Seiten
def OptimizeAllSides(cube):
    cube1 = PerlenOptimieren(cube)  # Cube mit erstem Optimierungsverfahren optimieren
    cube2 = np.rot90(cube, 1)  # Cube ein Mal um 90 Grad im Gegenuhrzeigersinn rotieren
    cube2 = PerlenOptimieren(cube2)  # Cube von rechts durch das Optimierungsverfahren lassen
    cube3 = np.rot90(cube, 2)  # Cube zwei Mal rotieren
    cube3 = PerlenOptimieren(cube3)  # Von hinten durchlassen
    cube4 = np.rot90(cube, 3)  # Cube drei Mal rotieren
    cube4 = PerlenOptimieren(cube4)  # Von links durch PerlenOptimieren lassen.

    sz = cube.shape
    for z in range(sz[2]):  # Für jede Ebene in allen vier Würfeln die Anzahl Punkte zusammenzählen
        a = np.sum(cube1[:, :, z])
        b = np.sum(cube2[:, :, z])
        c = np.sum(cube3[:, :, z])
        d = np.sum(cube4[:, :, z])
        min_index = np.argmin([a, b, c, d])  # Die Ebene mit der kleinsten Anzahl Punkte heraussuchen
    return(cube1)


#################################################################
# Funktion CalculateMin (cube)
#   Input: cube - dreidimensionales NumPy-Array der Projektionen durch den Würfel
#   Output: Die Anzahl Punkte im Optimum
#   Beschreibung: Berechnet die minimal mögliche Anzahl Punkte, um diese zwei Motive darzustellen.
def CalculateMin (cube):
    print("Die Minima der Ebenen sind:")
    if imageC is not None:
        print("Für Würfel mit 3 Seiten kann keine minimale Anzahl Punkte berechnet werden.")
    else:
        sz = cube.shape
        summe = 0
        for z in range(sz[2]):  # Für jede Ebene
            # Entlang der Axis 0 in allen Spalten die Summe der Punkte zusammengezählen
            a = np.sum(cube[:, :, z], 0)
            # Entlang der Axis 1 in allen Zeilen die Summe der Punkte zusammengezählen
            b = np.sum(cube[:, :, z], 1)

            c = np.size(np.where(a > 0))  # Die Anzahl Spalten zusammenzählen, in denen a > 0
            d = np.size(np.where(b > 0))  # Die Anzahl Zeilen zusammenzählen in denen b > 0

            if c > d:  # Je nachdem ob c oder d grösser ist, entspricht dies der minimalen möglichen Anzahl Punkte.
                print("Ebene:", z, ":", c)
                summe = summe+c

            else:
                print("Ebene", z, ":", d)
                summe = summe + d
        print("Das Optimum besteht aus", summe, "Punkten.")
        return (summe)


#################################################################
# Funktion PlotMinCube (cube)
#   Input: cube - dreidimensionales NumPy-Array der Projektionen durch den Würfel
#   Output: Optimierter cube
#   Beschreibung: Minimiert die Anzahl geplotteter Punkte um durchschnittliche 95,68 %.

def PlotMinCube (cube):
    sz = cube.shape
    if imageC is not None:
        print("Für Würfel mit 3 Seiten kann keine minimale Anzahl Punkte dargestellt werden.")
    else:
        optimum = np.zeros(sz)  # Eine leere Kopie des Würfels erzeugen
        for z in range(sz[2]):  # Für jede Ebene
            # Entlang der Axis 0 wird in allen Spalten die Summe der Punkte zusammengezählt.
            a = np.sum(cube[:, :, z], 0)
            # Entlang der Axis 1 wird in allen Zeilen die Summe der Punkte zusammengezählt.
            b = np.sum(cube[:, :, z], 1)

            c = np.size(np.where(a > 0))  # Die Anzahl Spalten zusammenzählen, in denen a > 0
            d = np.size(np.where(b > 0))  # Die Anzahl Zeilen zusammenzählen in denen b > 0

            if d >= c:  # Beginnend bei der längeren Seite

                aa = np.zeros(a.shape)  # Einen leeren String erstellen
                for x in range(sz[0]):  # Für alle Zeilen in die x Richtung
                    if b[x] > 0:  # Wenn es in dieser Zeile einen Punkt hat,
                        pointSetFlag = False

                        for y in range(sz[1]):  # Für jede einzelne Position dieser Zeile
                            if (cube[x, y, z] > 0) and (aa[y] == 0):  # Wenn es dort einen Punkt hat
                                # und in derselben Spalte noch kein Punkt gesetzt wurde,
                                optimum[x, y, z] = 1  # so wird in leerer Würfelkopie an dieser Stelle ein Punkt gesetzt
                                aa[y] = +1  # Im String wird an der Stelle dieser Spalte gespeichert,
                                # dass dort ein Punkt gesetzt wurde.
                                pointSetFlag = True  # True = Ein Punkt wurde gesetzt
                                break  # Sobald ein Punkt gesetzt wird, die Zeile verlassen

                        if pointSetFlag == False:  # Wenn nach dem Durchgehen der Zeile noch kein Punkt gesetzt wurde,
                            for y in range(sz[1]):  # so wird die Zeile noch einmal durchgegangen
                                if (cube[x, y, z] > 0): # Und an der erst möglichen Stelle ein Punkt gesetzt
                                    optimum[x, y, z] = 1
                                    aa[y] = +1
                                    break
                            print()

            else:  # Wenn c > d

                bb = np.zeros(b.shape)  # Einen leeren String erstellen
                for y in range(sz[1]):  # Für alle Spalten in die y Richtung
                    if a[y] > 0:  # Wenn es in dieser Spalte einen Punkt hat,
                        pointSetFlag = False

                        for x in range(sz[0]):  # Für jede einzelne Position dieser Spalte
                            if (cube[x, y, z] > 0) and (bb[x] == 0):  # Wenn es dort einen Punkt hat
                                # und in derselben Zeile noch kein Punkt gesetzt wurde,
                                optimum[x, y, z] = 1  # so wird in leerer Würfelkopie an dieser Stelle ein Punkt gesetzt
                                bb[x] = +1  # Im String wird an der Stelle dieser Zeile gespeichert,
                                # dass dort ein Punkt gesetzt wurde.
                                pointSetFlag = True  # True = Ein Punkt wurde gesetzt
                                break  # Sobald ein Punkt gesetzt wird, Spalte verlassen

                        if pointSetFlag == False:  # Wenn nach Durchgehen der Spalte noch kein Punkt gesetzt wurde,
                            for x in range(sz[0]):  # so wird die Spalte noch einmal durchgegangen
                                if (cube[x, y, z] > 0):  # und an der erst möglichen Stelle ein Punkt gesetzt.
                                    optimum[x, y, z] = 1
                                    bb[x] = +1
                                    break
        cube = optimum
        return(cube)


#################################################################
# Funktion RandomPlotMinCube (cube)
#   Input: cube - dreidimensionales NumPy-Array der Projektionen durch den Würfel
#   Output: Optimierter cube mit einer willkürlichen Verteilung der Punkte
#   Beschreibung: Randomisiert die Verteilung der Punkte im Optimum
def RandomPlotMinCube (cube):
    if imageC is not None:
        print("Für Würfel mit 3 Seiten kann keine minimale Anzahl Punkte berechnet und dargestellt werden.")
    else:
        sz = cube.shape
        optimum = np.zeros(sz)  # Eine leere Kopie des Würfels erzeugen
        for z in range(sz[2]):  # Für jede Ebene
            # Entlang der Axis 0 wird in allen Spalten die Summe der Punkte zusammengezählt
            a = np.sum(cube[:, :, z], 0)
            # Entlang der Axis 1 wird in allen Zeilen die Summe der Punkte zusammengezählt
            b = np.sum(cube[:, :, z], 1)

            c = np.size(np.where(a > 0))  # Die Anzahl Spalten zusammenzählen, in denen a > 0
            d = np.size(np.where(b > 0))  # Die Anzahl Zeilen zusammenzählen in denen b > 0

            if d >= c:  # Beginnend bei der längeren Seite
                aa = np.zeros(a.shape)  # Einen leeren String erstellen
                for x in range(sz[0]):  # Für alle Zeilen in die x Richtung
                    if b[x] > 0:  # Wenn es in dieser Zeile einen Punkt hat,
                        pointlist = np.array([], dtype='int')  # Eine leere Liste erzeugen
                        for y in range(sz[1]):  # Alle Punkte in dieser Zeile durchgehen
                            if cube[x, y, z] > 0:  # Wenn es im Originalcube dort einen Punkt hat,
                                pointlist = np.append(pointlist, y)  # so wird diese y-Stelle in die Liste aufgenommen.

                        np.random.shuffle(pointlist)  # Elemente der Liste willkürlich mischen
                        pointSetFlag = False
                        for y in list(pointlist):  # Jedes Element der Liste durchgehen.
                            if aa[y] == 0:  # Wenn es in dieser Spalte noch keinen anderen Punkt hat,
                                optimum[x, y, z] = 1  # so wird der Punkt in die Würfelkopie gesetzt.
                                aa[y] = +1  # Damit in Zukunft keine Punkte mehr in diese Spalte gesetzt werden,
                                # wird an dieser Stelle im aa eine 1 gesetzt.
                                pointSetFlag = True  # True = Ein Punkt wurde gesetzt
                                break  # Sobald ein Punkt gesetzt wird, wird die Zeile verlassen

                        if pointSetFlag == False:  # Wurde kein Punkt gesetzt, so gab es in jeder Spalte bereits einen Punkt
                            y = pointlist[0]  # Dann ist egal, wo der Punkt gesetzt wird, nehmen wir einfach [0]
                            optimum[x, y, z] = 1

            else:  # Der gleiche Prozess wie in der if Schlaufe, von der anderen Seitenlänge her
                bb = np.zeros(b.shape)
                for y in range(sz[1]):
                    if a[y] > 0:
                        pointlist = np.array([], dtype='int')
                        for x in range(sz[0]):
                            if cube[x, y, z] > 0:
                                pointlist = np.append(pointlist, x)

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
        cube = optimum
        return(cube)


#################################################################
# Funktion CalcRandom(cube)
#   Input: cube - dreidimensionales NumPy-Array der Projektionen durch den Würfel
#   Output: Anzahl Möglichkeiten
#   Beschreibung: Berechnet die Anzahl Möglichkeiten, um das Optimum dieser beiden Motive darzustellen.
def CalcRandom (cube):
    if imageC is not None:
        print("Für Würfel mit 3 Seiten kann keine minimale Anzahl Punkte berechnet und dargestellt werden.")
    else:
        sz = cube.shape
        pointlist = np.array([], dtype='int')  # Eine leere Liste erzeugen
        for z in range(sz[2]):  # Für jede Ebene
            # Entlang der Axis 0 wird in allen Spalten die Summe der Punkte zusammengezählt.
            a = np.sum(cube[:, :, z], 0)
            # Entlang der Axis 1 wird in allen Zeilen die Summe der Punkte zusammengezählt.
            b = np.sum(cube[:, :, z], 1)

            c = np.size(np.where(a > 0))  # Die Anzahl Spalten zusammenzählen, in denen a > 0
            d = np.size(np.where(b > 0))  # Die Anzahl Zeilen zusammenzählen in denen b > 0

            if d >= c:  # c = Breite des Schnittflächenrechtecks, d = Länge
                möglichkeiten = math.factorial(c) * c ^ (d-c)  # Formel der Anzahl Möglichkeiten pro Ebene
                pointlist = np.append(pointlist, float(möglichkeiten))  # Anzahl Möglichkeiten einer Liste hinzufügen
            else:  # d = Breite, c = Länge
                möglichkeiten = math.factorial(d) * d ^ (c-d)  # Formel der Anzahl Möglichkeiten pro Ebene
                pointlist = np.append(pointlist, float(möglichkeiten))  # Anzahl Möglichkeiten in Liste speichern
            print(möglichkeiten)  # Anzahl Möglichkeiten pro Ebene ausgeben
        total = np.prod(pointlist)  # Anzahl Möglichkeiten der einzelnen Ebenen miteinander multiplizieren
        print("Diese Modell hat", total, "Möglichkeiten, um das Optimum darzustellen.")


#################################################################
# Funktion Modellbau(cube)
#   Input: cube - Das dreidimensionale NumPy-Array der Motive
#   Output: Visualisierung der einzelnen Ebenen. X = Ein Punkt .= Kein Punkt
#   Beschreibung: Gibt einen graphischen Bauplan für die Perlenmodelle aus.
def Modellbau(cube):
    for x in range(cube.shape[0]-1, -1, -1):  # Für jede Ebene in einem gekippten Würfel
        # (Ebenen werden von hinten nach vorne gespannt im Perlenmodell)
        Ebene = ""  # Leeren string "Ebene" erstellen
        for y in range(cube.shape[1]):
            for z in range(cube.shape[2]):  # Alle Kombinationen von y und z Koordinaten in cube abchecken
                if cube[x, y, z] == 1:  # Ist dort ein Punkt, so wird X dem String "Ebene" hinzugefügt
                    Ebene += " X"
                else:
                    Ebene += " ."  # Kein Punkt, so wird . dem String "Ebene" hinzugefügt
            Ebene += "\n"  # Neue Linie hinzufügen für die nächste Zeile
        print(Ebene)  # Jede Ebene ausgeben


############################################################
# Funktion Drahtmodel
#   Input: cube - Das dreidimensionale NumPy-Array der Motive
#   Output: Visualisierung des auf Drahtbreite reduzierten Modells
#   Beschreibung: Reduziert das Modell auf sein Skelett.
def skeletonizeCube(cube):
    cube = skeletonize(cube)
    return(cube)

if __name__ == "__main__":  # Damit der untere Teil nicht ausgeführt wird, wenn dieses File als Modul importiert wird

    mnist = MNIST()  # MNIST Datensatz laden
    if imageC is not None: # Wenn 3 Bilder gegeben sind:

        if type(imageA) == int:  # Ist imageA eine Ziffer, wird das zugehörige Bild aus MNIST laden
            img1 = MnistImage(imageA)
        else:
            img1 = LoadImage(imageA, threshold_imageA) # Else, das zugehörige Bild vom Filelink laden

        if type(imageB) == int:
            img2 = MnistImage(imageB)
        else:
            img2 = LoadImage(imageB,threshold_imageB)

        if type(imageC) == int:
            img3 = MnistImage(imageC)
        else:
            img3 = LoadImage(imageC,threshold_imageC)

        img1 = np.rot90(img1, -1, (0, 1))  # img1 rotieren
        vol2 = np.tile(img1, (img1.shape[0], 1, 1))  # img1 so oft aufeinanderstappeln wie das Bild lang ist

        img2 = np.rot90(img2, -1, (0, 1))  # img2 rotieren
        vol1 = np.tile(img2, (img2.shape[0], 1, 1))  # img2 so oft aufeinanderstappeln wie das Bild lang ist
        vol1 = np.rot90(vol1, 1)  # vol2 um 90 Grad rotieren, um das Bild von einer anderen Seite als vol1 darzustellen

        img3 = np.rot90(img3, -1, (0,1)) # img3 rotieren
        vol3 = np.tile(img3, (img3.shape[0], 1, 1))  # img3 so oft aufeinanderstappeln wie das Bild lang ist
        vol3 = np.rot90(vol3, 1, (0,2))  # vol3 um 90 Grad rotieren, um das Bild von einer anderen Seite als vol1 darzustellen

        ### Start Kontrollfunktion
        # Kontrollieren, ob die drei Motive kompatibel miteinander sind

        # Entlang der Axis 1 wird die Summe der Punkte in allen Zeilen zusammengezählt.
        a = np.sum(img1, 1)
        h = np.sum(img1,0)

        # Entlang der Axis 1 wird die Summe der Punkte in allen Zeilen zusammengezählt.
        b = np.sum(img2, 1)
        f = np.sum(img2,0)

        # Entlang der Axis 1 wird die Summe der Punkte in allen Zeilen zusammengezählt.
        e = np.sum(img3, 1)
        g = np.sum(img3,0)

        c = np.sign(a)  # Alle Werte > 0 in eine 1 umwandeln, die anderen Werte in eine 0
        d = np.sign(b)
        e = np.sign(e)
        f = np.sign(f)
        g = np.sign(g)

        # Wenn eine Ansicht, leere Ebenen hat und somit einen Teil der anderen Ansicht auslöscht
        if not np.all(c == d):
            print("Die Ansicht ist unvollständig, denn die Höhe von img1 und img2 ist nicht identisch.")
        if not np.all(h == g):
            print ("Die Ansicht ist unvollständig, denn die Breite von img1 ist nicht identisch mit der Breite von img3.")
        if not np.all(e == f):
            print("Die Ansicht ist unvollständig, denn die Breite von img2 ist nicht identisch mit der Höhe von img3.")
        # Ende Kontrollfunktion

        cube = vol1 * vol2 * vol3 # Würfel der drei Bilder miteinander multiplizieren

    else:  # Wenn nur zwei Bilder gegeben sind:

        if type(imageA) == int:  # Ist imageA eine Ziffer, wird das zugehörige Bild aus MNIST laden
            img1 = MnistImage(imageA)
        else:
            img1 = LoadImage(imageA,threshold_imageA) # Else, das zugehörige Bild vom Filelink laden

        if type(imageB) == int:
            img2 = MnistImage(imageB)
        else:
            img2 = LoadImage(imageB, threshold_imageB)

        img1 = np.rot90(img1, -1, (0, 1))  # img1 rotieren
        vol2 = np.tile(img1, (img1.shape[0], 1, 1))  # img1 so oft aufeinanderstappeln wie das Bild lang ist

        img2 = np.rot90(img2, -1, (0, 1))  # img2 rotieren
        vol1 = np.tile(img2, (img2.shape[0], 1, 1))  # img2 so oft aufeinanderstappeln wie das Bild lang ist
        vol1 = np.rot90(vol1, 1)  # vol2 um 90 Grad rotieren, um das Bild von einer anderen Seite als vol1 darzustellen

        ### Start Kontrollfunktion
        # Kontrollieren, ob die beiden Motive kompatibel miteinander sind

        # Entlang der Axis 1 wird die Summe der Punkte in allen Zeilen zusammengezählt.
        a = np.sum(img1, 1)
        # Entlang der Axis 1 wird die Summe der Punkte in allen Zeilen zusammengezählt.
        b = np.sum(img2, 1)
        c = np.sign(a)  # Alle Werte > 0 in eine 1 umwandeln, die anderen Werte in eine 0
        d = np.sign(b)

        # Wenn eine Ansicht, leere Ebenen hat und somit einen Teil der anderen Ansicht auslöscht
        if not np.all(c == d):
            print("Die Ansicht ist unvollständig.")
        # Ende Kontrollfunktion

        cube = vol1 * vol2 # Würfel der beiden Bilder miteinander multiplizieren


    cubegraph = cube > 0  # Nur Punkte darstellen, in denen eine Zahl > 1 ist (denn 0 = weiss , 1 = schwarz).
    cube[cubegraph] = 1.  # Alle Werte, die true sind in cubegraph, in eine 1 umwandeln

    # Konfigurierter Optimierungsalgorithmus anwenden
    if Optimierungsalgorithmus == 0:
        cube = cube
    elif Optimierungsalgorithmus == 1:
        cube = PerlenOptimieren(cube)
    elif Optimierungsalgorithmus == 2:
        cube = OptimizeAllSides(cube)
    elif Optimierungsalgorithmus == 3:
        cube = PlotMinCube(cube)
    elif Optimierungsalgorithmus == 4:
        cube = RandomPlotMinCube(cube)
    else:
        print("Ungültige Eingabe für den Optimierungsalgorithmus")

    if PrintBauplan == True:
        Modellbau(cube)

    if AnzahlPunkte == True:
        anzahlperlen = np.size(np.where(cube == True)) / 3  # Anzahl geplotteter Punkte zählen.
        # /3 , da np.where jeden Punkt mit drei Koordinaten angibt
        print("Die Anzahl Perlen ist:", anzahlperlen)

    if MinimalePunkte == True:
        CalculateMin(cube)

    if AnzahlMöglichkeiten == True:
        CalcRandom(cube)

    if Drahtmodel == True:
        cube = skeletonizeCube(cube)

    # Das Modell darstellen:
    # Meshgrid für die x, y, z Koordinaten erzeugen
    cube = cube > 0

    x, y, z = np.meshgrid(np.arange(cube.shape[0]),
                          np.arange(cube.shape[1]),
                          np.arange(cube.shape[2]))

    fig = plt.figure()  # 3D Plot erstellen
    ax = fig.add_subplot(111, projection='3d')  # Plot in Feld 111 tun

    ax.scatter(x[cube], y[cube], z[cube])  # Punkte im Cube plotten

    # Achsen anschreiben
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()

    # Plot darstellen
    plt.title('Modell')
    plt.show()


    if STL == True:  # NumPy-Array in ein STL-File umwandeln
        # Schicht 0 an jeder Seite von cube hinzufügen, damit die Aussenfläche im STL-File inbegriffen ist
        cubepadded = np.pad(cube, pad_width=1, mode='constant', constant_values=0)
        verts, faces, normals, values = measure.marching_cubes(cubepadded, spacing=(1, 1, 1))

        surf = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))   # Das Mesh erzeugen
        for i, f in enumerate(faces):
            for j in range(3):
                surf.vectors[i][j] = verts[f[j], :]

        surf.save('try.stl')  # Als STL-File unter dem Namen 'try.stl' abspeichern