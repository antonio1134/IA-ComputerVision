import cv2

# Inicializa la cámara
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()  # Capturar fotograma de la cámara
    if not ret:
        print("Error al capturar el fotograma.")
        break

    # Procesar el fotograma
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 10, 150)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        # Identificar la forma
        if len(approx) == 3:
            cv2.putText(frame, 'Triangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:  # Tolerancia para cuadrados
                cv2.putText(frame, 'Cuadrado', (x, y - 5), 1, 1, (0, 255, 0), 1)
            else:
                cv2.putText(frame, 'Rectangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 5:
            cv2.putText(frame, 'Pentagono', (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 6:
            cv2.putText(frame, 'Hexagono', (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) > 10:
            cv2.putText(frame, 'Circulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    # Mostrar el fotograma procesado
    cv2.imshow('Deteccion en Tiempo Real', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
