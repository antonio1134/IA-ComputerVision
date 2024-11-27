import cv2
import numpy as np
import matplotlib.pyplot as plt

captura = cv2.VideoCapture(0)

while True:
    ret, frame = captura.read()
    
    if not ret:
        break
    
    cv2.imshow('Video en vivo', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
captura.relase()
cv2.destroyAllWindows()
