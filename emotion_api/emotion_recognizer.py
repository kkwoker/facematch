import cv2
import sys
import json
import time
import dlib
import numpy as np
from imutils import face_utils

# Make tensorflow silent SSE4.1 SSE4.2 AVX, AVX2, FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import model_from_json

def load_face_detector_cv():
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

def load_landmark_detector():
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    return predictor

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
def get_face_image_cv(camera_image, face_detector):
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

# Returns (face_image, (x, y, w, h))
def get_face_image_dlib(camera_image, face_detector):
    img_gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY, 1)
    faces = face_detector(img_gray, 0)

    # Crop the first face found
    if len(faces):
        face = faces[0]
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_image = img_gray[y:y + h, x:x + w]
        return (face_image, (x, y, w, h))

    return (None, None)


def draw_face_landmark_lines(camera_image, face_landmarks, top_left, color, loop=False):
    for i in range(1, len(face_landmarks)):
        x1, y1 = face_landmarks[i - 1]
        x2, y2 = face_landmarks[i]
        point1_xy = (top_left[0] + x1, top_left[1] + y1)
        point2_xy = (top_left[0] + x2, top_left[1] + y2)
        cv2.line(camera_image, point1_xy, point2_xy, color)

    if loop:
        x1, y1 = face_landmarks[len(face_landmarks) -1]
        x2, y2 = face_landmarks[0]
        point1_xy = (top_left[0] + x1, top_left[1] + y1)
        point2_xy = (top_left[0] + x2, top_left[1] + y2)
        cv2.line(camera_image, point1_xy, point2_xy, color)

def draw_face_landmarks(camera_image, face_landmarks, top_left, color):
    outline = face_landmarks[0:17]
    left_brow = face_landmarks[17:22]
    right_brow = face_landmarks[22:27]
    nose = face_landmarks[27:31]
    nostrils = face_landmarks[30:36]
    right_eye = face_landmarks[36:42]
    left_eye = face_landmarks[42:48]
    outer_lip = face_landmarks[48:60]
    inner_lip = face_landmarks[60:68]

    draw_face_landmark_lines(camera_image, outline, top_left, color)
    draw_face_landmark_lines(camera_image, left_brow, top_left, color)
    draw_face_landmark_lines(camera_image, right_brow, top_left, color)
    draw_face_landmark_lines(camera_image, nose, top_left, color)
    draw_face_landmark_lines(camera_image, nostrils, top_left, color, True)
    draw_face_landmark_lines(camera_image, left_eye, top_left, color, True)
    draw_face_landmark_lines(camera_image, right_eye, top_left, color, True)
    draw_face_landmark_lines(camera_image, outer_lip, top_left, color, True)
    draw_face_landmark_lines(camera_image, inner_lip, top_left, color, True)

    # for i in range(len(face_landmarks)):
    #     x, y = face_landmarks[i]
    #     point_xy = (top_left[0] + x, top_left[1] + y)
    #     cv2.putText(camera_image, str(i + 1), point_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
    #     # cv2.circle(camera_image, point_xy, 1, color, -1)

def recognize_emotions(camera_image, debug=False):
    #face_image, rect = get_face_image_dlib(camera_image, face_detector_dlib)
    face_image, rect = get_face_image_cv(camera_image, face_detector_cv)
    if face_image is None: return (None, None)# No face detected
    emotions = predict_emotion(face_image, emotion_model)

    top_left = rect[0:2]
    bottom_right = (rect[0] + rect[2], rect[1] + rect[3])

    # Facial landmark shapes
    face_landmarks = landmark_detector(face_image, dlib.rectangle(0, 0, rect[2], rect[3]))
    face_landmarks = face_utils.shape_to_np(face_landmarks)

    # debug only
    if debug:
        green_color = (0, 255, 0)
        blue_color = (255, 0, 0)
        red_color = (0, 0, 255)

        # draw face and rect
        cv2.rectangle(camera_image, top_left, bottom_right, green_color, 1)
        draw_face_landmarks(camera_image, face_landmarks, top_left, red_color)

        # draw emotion values
        text_top_left = (top_left[0] + 10, top_left[1] + 20)
        text_height = 20
        max_line_width = rect[2] * 1.5

        for i in range(len(emotions)):
            emotion_name, emotion_score = emotions[i]
            emotion_xy = (text_top_left[0], text_top_left[1] + (i * text_height))
            cv2.putText(camera_image, emotion_name, emotion_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color)
            line_xy = (emotion_xy[0], emotion_xy[1] + 5)
            line_width = int(max_line_width * emotion_score)
            cv2.line(camera_image, line_xy, (line_xy[0] + line_width, line_xy[1]),  blue_color, 2)

        # Show final annotated images
        cv2.imshow('camera_image', camera_image)
        cv2.waitKey(1)

    return (emotions, rect)


face_detector_dlib = dlib.get_frontal_face_detector()
face_detector_cv = load_face_detector_cv()
landmark_detector = load_landmark_detector()
emotion_model = load_emotion_model()

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        recognize_emotions(frame, debug=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
