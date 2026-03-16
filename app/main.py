import cv2
import numpy as np
import torch
from model_skeleton import CNN_Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN_Model(input_shape=1, hidden_layers=10, output_shape=10).to(device)
model.load_state_dict(torch.load('model/MNIST_model.pth', map_location=device))
model.eval()

cap = cv2.VideoCapture('https://192.168.0.104:8080/video')

widthImg = 720
heightImg = 720


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    imgClosed = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
    # imgDilate = cv2.dilate(imgThresh, kernel, iterations=1)
    # imgErode = cv2.erode(imgDilate, kernel=kernel, iterations=1)

    return imgClosed


def getPredictions(img):
    img = cv2.resize(img, (28, 28))

    img = img.astype(np.float32) / 255.0

    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(img)

        softmax = torch.softmax(output, dim=1)

        confidence, prediction_tensor = torch.max(softmax, dim=1)

        prediction = prediction_tensor.item()
        prob_value = confidence.item()

    return prediction, prob_value


def findContours(img):
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 700:
            x, y, w, h = cv2.boundingRect(cnt)

            aspectRatio = float(w) / h

            if 0.2 < aspectRatio < 1.0:
                imgNumber = img[y:y+h, x:x+w]

                number, confidence = getPredictions(imgNumber)

                if confidence > 0.7:
                    cv2.rectangle(imgContour, (x, y),
                                  (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(
                        imgContour, f'{number} ({int(confidence * 100)}%)', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)


while cap.isOpened():
    succ, img = cap.read()

    if not succ or img is None:
        print('Unable to open Video!!')
        break

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    # Preprocessing
    imgPreProcessed = preProcessing(img)

    # Find Contours
    findContours(imgPreProcessed)

    cv2.imshow('Press ESC to exitt', imgPreProcessed)
    cv2.imshow('Press ESC to exit', imgContour)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
