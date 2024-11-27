import cv2

captura = cv2.VideoCapture(0)
ret, frame = captura.read()

if ret:
    cv2.imwrite('/home/antonio/9oSemestre/IA/u4/ciculos.jpg', frame)
    
captura.release()
