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

class EmotionHandler(tornado.web.RequestHandler):
  def get(self):
    print('get')
    self.write("oh hai! you need to make a post request")

  def post(self):
    print("POST %s from %s" % (self.request.path, self.request.remote_ip))
    self.set_header('Content-Type', 'application/json')

    # Error check the image
    jpg_image = None
    emotion = None
    emotion_score = 0
    emotion_score_map = {}

    try:
      post_files = self.request.files
      emotion = self.get_argument('emotion', default=None)
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
    emotion_scores, face_rect = emotion_recognizer.recognize_emotions(camera_image, debug=True)

    # Build emotion score map
    if emotion_scores:
      for emotion_name, emotion_score in emotion_scores:
        emotion_score_map[emotion_name] = int(emotion_score * 10)
      if not emotion: # get the top emotion
        emotion = emotion_scores[0][0]
      emotion_score = emotion_score_map[emotion]

    result = dict(emotion=emotion, score=emotion_score)
    self.write(json.dumps(result))
    if face_rect:
      print(result)
      print(emotion_score_map)

if __name__ == '__main__':
  port = 8080
  application = tornado.web.Application([
    (r'/emotion', EmotionHandler),
  ])
  application.listen(port=port)
  print('Serving emotion_api on %s' % port)
  tornado.ioloop.IOLoop.instance().start()
