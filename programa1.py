import cv2
import numpy as np
import matplotlib.pyplot as plt
#Leer imagen
imagen = cv2.imread('/home/antonio/9oSemestre/IA/u4/ciculos.jpg')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(gris, 100, 200)
#Mostrar imagen
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen color gris', gris)
cv2.imshow('Imagen gris bordes', bordes)
#Espera a que se presione una tecla
cv2.waitKey(0)
#Cierra la ventana
cv2.destroyAllWindows()
