import io, os
from typing import List
import numpy as np
import requests
import torch
from PIL import Image
import datetime
import warnings

warnings.filterwarnings("ignore")

POSE_NAME = ['AP', 'LAT']

yolo = torch.hub.load('',
                           'custom',
                           path='model/best.pt',
                           source='local')
yolo.cpu()



def detect(image):
  print("uni")
  result = yolo(image, size=416)

  con    = 0
  x_pos  = -1
  y_pos  = -1

  if len(result.pandas().xyxy[0]) > 0:
      data = result.pandas().xyxy[0]

      for i, j in data.iterrows():
              con = j['confidence']
              x   = j['xmin']
              y   = j['ymin']
              w   = j['xmax']
              h   = j['ymax']



