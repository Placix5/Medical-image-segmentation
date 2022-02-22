import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

############################################################################################################################################################################

def Segmentation(img_path):

    # Cargamos la imagen que vamos a analizar en dos variables, la primera para realizar las diferentes transformaciones y la segunda de ellas
    # para poder mostrarla al final y poder comparar visualmente
    img = cv.imread(img_path)
    img_original = cv.imread(img_path)

    # Convertimos la imagen a escala de grises para poder realizar el tratamiento de una forma sencilla en un único canal
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Aplicamos un filtro Threshold con lo que buscamos poder diferenciar el fondo de la imagen. Este filtro define un umbral de color, todos aquellos píxeles
    # que no cumplan dicho umbral, se les proporciona un valor y aquellos que sí lo cumplen se les proporciona otro
    ret, thresh = cv.threshold(img_grey, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # Definimos un kernel para utilizarlo en la operación morfológica siguiente
    kernel = np.ones((3,3), np.uint8)
    # Aplicamos una operación de apertura para eliminar el ruido de la imagen
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)

    # Extraemos la operación de dilatación para extraer el fondo seguro de la imagen
    sure_bg = cv.dilate(opening, kernel, iterations = 3)

    # Aplicamos una transformación por distancia a cada centro, de tal forma de que cuanto más alejado esté el pixel del centro, mayor será la probabilidad
    # de que ese área sea fondo. De esta forma conseguimos extraer lo que no es fondo de la imagen
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    # Aplicamos un filtro Threshold para extraer el no fondo
    ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),255,0)

    # Convertimos el no fondo para poder realizar operaciones con él
    sure_fg = np.uint8(sure_fg)
    # Extraemos las zonas desconocidas eliminandole al fondo la zona que sabemos que no es fondo
    unknow = cv.subtract(sure_bg, sure_fg)

    # Generamos los marcadores a partir de las componentes conexas de la imagen
    ret, markers = cv.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknow == 255] = 0

    # Aplicamos el algoritmo WaterShed a la imagen con los marcadores como semillas
    markers = cv.watershed(img, markers)

    # Dibujamos el contorno de color azul
    img[markers == -1] = [255,0,0]

    # Mostramos la imagen original y la imagen con el contorno dibujado
    cv.imshow("Imagen original",img_original)
    cv.imshow("Imagen watershed",img)
    cv.waitKey(0)

############################################################################################################################################################################

# Para hacerlo modular, se ha encapsulado el código en una función, de esta forma podemos llamarlo cuantas veces queramos
# definiendo únicamente el path de la imagen que queremos analizar
img_path = "Images/A_T2_Gel1.tif"
Segmentation(img_path)