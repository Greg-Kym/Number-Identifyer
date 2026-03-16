import cv2

cap = cv2.VideoCapture(0)

widthImg = 720
heightImg = 720

while cap.isOpened():
    succ, img = cap.read()

    if not succ or img is None:
        print('Unable to open Video!!')
        break

    cv2.imshow('Press ESC to exit', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
