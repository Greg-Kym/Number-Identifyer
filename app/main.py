import cv2

cap = cv2.VideoCapture('https://192.168.0.104:8080/video')

widthImg = 720
heightImg = 720


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    _, imgThresh = cv2.threshold(imgBlur, 127, 255, cv2.THRESH_BINARY_INV)

    return imgThresh


while cap.isOpened():
    succ, img = cap.read()

    if not succ or img is None:
        print('Unable to open Video!!')
        break

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (widthImg, heightImg))

    ###### Preprocessing ######
    imgPreProcessed = preProcessing(img)

    cv2.imshow('Press ESC to exit', imgPreProcessed)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
