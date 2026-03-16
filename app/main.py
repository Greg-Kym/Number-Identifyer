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
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 2)
    imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 25, 10)
    kernel = np.ones((5, 5), np.uint8)
    # imgClosed = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
    imgDilate = cv2.dilate(imgThresh, kernel, iterations=2)
    imgErode = cv2.erode(imgDilate, kernel=kernel, iterations=1)

    return imgErode


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

        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)

            aspectRatio = float(w) / h

            if 0.1 < aspectRatio < 1.5:

                # Ensure we don't crop outside the image boundaries
                max_dim = max(w, h)
                pad = 10

                # Calculate center of the detected number
                centerX, centerY = x + w // 2, y + h // 2

                # Create a square bounding box
                x1 = max(0, centerX - max_dim // 2 - pad)
                y1 = max(0, centerY - max_dim // 2 - pad)
                x2 = min(widthImg, centerX + max_dim // 2 + pad)
                y2 = min(heightImg, centerY + max_dim // 2 + pad)

                imgNumber = img[y1:y2, x1:x2]

                number, confidence = getPredictions(imgNumber)

                if confidence > 0.20:
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
