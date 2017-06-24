import cv2
import sys
import json
import time
import numpy as np

# Make tensorflow silent SSE4.1 SSE4.2 AVX, AVX2, FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import model_from_json

def load_face_detector():
    haarcascade_path = 'haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(haarcascade_path)
    return face_detector

def load_emotion_model():
    # load json and create model from .json and load the h5 weights
    with open('model.json', 'r') as f:
        print("Loading model")
        model = model_from_json(f.read())
        model.load_weights('model.h5')
        print("Model loaded")

    return model

# returns [(emotion, score), (emotion, score), ...]
def predict_emotion(face_image_gray, emotion_model):
    # Resize image to 48x48 in a single channel
    img_width = 48
    resized_img = cv2.resize(face_image_gray, (img_width, img_width), interpolation=cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, img_width, img_width)

    # Predict scores and return sorted (score, name) tuples
    emotion_names = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotion_scores = emotion_model.predict(image, batch_size=1, verbose=False)
    emotion_scores = emotion_scores.reshape(emotion_scores.shape[1]).tolist()
    emotion_predictions = list(zip(emotion_names, emotion_scores))
    emotion_predictions.sort(key=lambda i: i[1], reverse=True) # sort by score
    return emotion_predictions

# Returns (face_image, (x, y, w, h))
def get_face_image(camera_image, face_detector):
    img_gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY, 1)

    faces = face_detector.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Crop the first face found
    if len(faces):
        x, y, w, h = faces[0].tolist()
        face_image = img_gray[y:y + h, x:x + w]
        return (face_image, (x, y, w, h))

    return (None, None)

def recognize_emotions(camera_image, face_detector, emotion_model, debug=False):
    face_image, rect = get_face_image(camera_image, face_detector)
    if face_image is None: return (None, None)# No face detected
    emotions = predict_emotion(face_image, emotion_model)

    # debug only
    if debug:
        cv2.imshow('Face', face_image)
        green_color = (0, 255, 0)
        top_left = rect[0:2]
        bottom_right = (rect[0] + rect[2], rect[1] + rect[3])

        # draw rectangle around face
        cv2.rectangle(camera_image, top_left, bottom_right, green_color, 2)

        # draw emotion values
        text_top_left = (top_left[0] + 10, top_left[1] + 20)
        text_height = 15
        max_line_width = rect[2]

        for i in range(len(emotions)):
            emotion_name, emotion_score = emotions[i]
            emotion_xy = (text_top_left[0], text_top_left[1] + (i * text_height))
            cv2.putText(camera_image, emotion_name, emotion_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color)
            line_xy = (emotion_xy[0], emotion_xy[1] + 2)
            line_width = int(max_line_width * emotion_score)
            cv2.line(camera_image, line_xy, (line_xy[0] + line_width, line_xy[1]), green_color)

        # Show final annotated images
        cv2.imshow('camera_image', camera_image)
        cv2.waitKey(1)

    return (emotions, rect)


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    face_detector = load_face_detector()
    emotion_model = load_emotion_model()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        recognize_emotions(frame, face_detector, emotion_model, debug=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
