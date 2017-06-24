#!/usr/bin/env python

from multiprocessing import Process, Queue
import os
import sys
import time
import tornado.ioloop
import tornado.web
import json
import cv2
import traceback
import numpy as np
import time
import emotion_recognizer

face_detector = emotion_recognizer.load_face_detector()
emotion_model = emotion_recognizer.load_emotion_model()

class EmotionHandler(tornado.web.RequestHandler):
  def get(self):
    print('get')
    self.write("oh hai! you need to make a post request")

  def post(self):
    print("POST %s from %s" % (self.request.path, self.request.remote_ip))
    self.set_header('Content-Type', 'application/json')

    # Error check the image
    jpg_image = None
    try:
      post_files = self.request.files
      jpg_image = post_files['file'][0]['body']
    except Exception:
      traceback.print_exc()

    if jpg_image is None:
      result = {'status': 'error', 'error_message': 'No image found'}
      self.write(json.dumps(result))
      return

    # Recognize the emotions
    np_arr = np.fromstring(jpg_image, np.uint8)
    camera_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    recognize = emotion_recognizer.recognize_emotions
    emotion_scores, face_rect = recognize(camera_image, face_detector, emotion_model, debug=True)
    emotion = None if emotion_scores is None else emotion_scores[0][0]
    result = dict(emotion=emotion, face_rect=face_rect, emotion_scores=emotion_scores)
    self.write(json.dumps(result))
    if emotion: print(result)

if __name__ == '__main__':
  application = tornado.web.Application([
    (r'/emotion', EmotionHandler),
  ])
  port = 8080
  address = '0.0.0.0'
  application.listen(port=port)
  print('Serving emotion_api on %s:%s' % (address, port))

  tornado.ioloop.IOLoop.instance().start()
