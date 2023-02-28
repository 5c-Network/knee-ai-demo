import os
import subprocess
# import psycopg2
# from psycopg2.extras import execute_values
# import sqlalchemy
import pandas as pd
import numpy as np
import urllib
import requests
import json
import shutil
# import boto3
import io
from PIL import Image
from datetime import date, timedelta
# import awscli
# from dotenv import load_dotenv
# import pydicom
# import pydicom as dicom
# import matplotlib.pyplot as plt
import PIL 
import torch
import warnings
import tensorflow as tf
import tensorflow_hub as hub
from io import BytesIO
from collections import defaultdict

# dotenv_path = '.env'
# load_dotenv(dotenv_path)
# warnings.filterwarnings("ignore")

# def connect( database ):
#     conn = psycopg2.connect(
#         database =  database,
#         user     =  os.environ.get("EUNAME"), 
#         password =  os.environ.get("EPASSWORD"), 
#         host     =  os.environ.get("EHOST"), 
#         port     =  os.environ.get("PORT")
#         )
#     cur = conn.cursor()
#     return cur, conn

# def download_files(path):
#     session = boto3.Session(
#         aws_access_key_id = os.environ.get("ACCESSKEY"), 
#         aws_secret_access_key = os.environ.get("SECRETACCESSKEY"), 
#         region_name = 'ap-south-1'
#     )
#     s3 = session.resource('s3')
#     bucket = s3.Bucket('5cnetwork-newserver-dicom')
    
#     image = bucket.Object(path)
#     img = image.get().get('Body').read()
#     res = pydicom.dcmread(io.BytesIO(img))
#     new_image = res.pixel_array.astype(float)
#     scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
#     scaled_image = np.uint8(scaled_image)

#     return Image.fromarray(scaled_image).convert('RGB')


# class S3ImagesInvalidExtension(Exception):
#     pass

# class S3ImagesUploadFailed(Exception):
#     pass

# class S3Images(object):
    
#     def __init__(self, aws_access_key_id, aws_secret_access_key, region_name):
#         self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, 
#                                      aws_secret_access_key=aws_secret_access_key, 
#                                      region_name=region_name)
        

#     def from_s3(self, bucket, key):
#         file_byte_string = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
#         return Image.open(BytesIO(file_byte_string))
    

#     def to_s3(self, img, bucket, key):
#         buffer = BytesIO()
#         img.save(buffer, self.__get_safe_ext(key))
#         buffer.seek(0)
#         sent_data = self.s3.put_object(Bucket=bucket, Key=key, Body=buffer)
#         if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
#             raise S3ImagesUploadFailed('Failed to upload image {} to bucket {}'.format(key, bucket))
        
#     def __get_safe_ext(self, key):
#         ext = os.path.splitext(key)[-1].strip('.').upper()
#         if ext in ['JPG', 'JPEG']:
#             return 'JPEG' 
#         elif ext in ['PNG']:
#             return 'PNG' 
#         else:
#             raise S3ImagesInvalidExtension('Extension is invalid') 


yolo = torch.hub.load('',
                          'custom',
                          path='model/od.pt',
                          source='local')
yolo.cpu()


def crop_knee(image):
  result = yolo(image, size=416)

  final = []
  if len(result.pandas().xyxy[0]) > 0:
      data = result.pandas().xyxy[0]
      for i, j in data.iterrows():
        if j['confidence'] > 0.50:
          final.append((image.crop((j['xmin'], j['ymin'], j['xmax'], j['ymax'])), j['confidence'] ))
  return final

pose = tf.keras.models.load_model(
       ('model/pose.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

def classify_view(image):
  temp = image.resize((224, 224))
  
  temp = np.array([np.array(temp)/255])
  res = pose.predict(temp, verbose = 0)
  cls = ['AP', 'Lateral']
  
  return cls[np.argmax(res)] , res[0][np.argmax(res)]


knee = tf.keras.models.load_model(
       ('model/model.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

def classify_knee(image):
  temp = image.resize((224, 224))  
  temp = np.array([np.array(temp)/255])
  res = knee.predict(temp, verbose = 0)
  cls = ['Abnormal', 'Normal']  
  return cls[np.argmax(res)] , res[0][np.argmax(res)]

def invert(image):
    image = image.convert('RGB') 
    x, y = image.size

    color_1 = defaultdict(int)
    for pixel in image.crop((0,y/2,50,(y/2)+50)).getdata():
        color_1[pixel] += 1

    color_2 = defaultdict(int)
    for pixel in image.crop((x-50,y/2,x,(y/2)+50)).getdata():
        color_2[pixel] += 1

    org_left  = max(color_1.items(), key=lambda a: a[1])[0][0]
    org_right = max(color_2.items(), key=lambda a: a[1])[0][0]

    # print(org_left, org_right)

    if org_left >= 150 or org_right >= 150:
        inverted_image = PIL.ImageOps.invert(image.convert('RGB') )

        color_1 = defaultdict(int)
        for pixel in inverted_image.crop((0,y/2,50,(y/2)+50)).getdata():
            color_1[pixel] += 1

        color_2 = defaultdict(int)
        for pixel in inverted_image.crop((x-50,y/2,x,(y/2)+50)).getdata():
            color_2[pixel] += 1

        # print(max(color_1.items(), key=lambda a: a[1])[0][0], max(color_2.items(), key=lambda a: a[1])[0][0])

        return inverted_image, True
    
    return image, False


# def main():
#     # cur, conn = connect("insight")

#     # tables = ['metadata_aug2021', 'metadata_dec2022', 'metadata_nov2022',
#     #     'metadata_jan2022', 'metadata_feb2022',
#     #     'metadata_mar2022', 'metadata_apr2022', 'metadata_may2022',
#     #     'metadata_jun2022', 'metadata_jul2022', 'metadata_aug2022',
#     #     'metadata_sep2022', 'metadata_oct2022', 'metadata_oct2021',
#     #     'metadata_nov2021', 'metadata_dec2021', 'metadata_sep2021']

#     # data = []
#     # for i in tables:
#     #     cur.execute(""" 
#     #             select 
#     #             study_id,
#     #             study_path, 
#     #             dicom_path, 
#     #             createdat 
#     #             from (
#     #             select  
#     #                 *,
#     #                 row_number() over(partition by study_id order by createdat ) rn
#     #             from 
#     #                 dicom.{0}  
#     #             where 
#     #                 bodypartexamined = 'KNEE' 
#     #             ) d
#     #             where 
#     #             rn = 1
#     #                 """.format(i))
#     #     data.append(pd.DataFrame(cur.fetchall()))

#     # data = pd.concat(data)

#     # cur.close()
#     # conn.close()

#     data = pd.read_csv('data.csv')

#     s3 = S3Images(os.environ.get("ACCESSKEY"), 
#          os.environ.get("SECRETACCESSKEY"), 
#          os.environ.get("REGIONNAME") )

#     cur, conn = connect("ai")

#     for i,j in data.iterrows():
#         data[i:].to_csv('data.csv', index = False)
#         try:
#             if i % 100 == 0:
#                 print(i)

#             path = '/'.join(j[2].split('/')[3:])
#             org  = download_files(path)
#             img, invert_res = invert(org)

#             cropped = crop_knee(img)
#             if cropped:
#                 flag = 0 
#                 for k in cropped:
#                     flag +=  1
#                     s3_path = 'knee-cropped-images/'+path+'_{0}.jpg'.format(flag)
#                     s3.to_s3(k[0], '5c-ai-image-data', s3_path)
#                     view, view_acc = classify_view(k[0])
#                     result = {'study_id': j[0],
#                                 'dicom_path': path,
#                                 'is_inverted': invert_res,
#                                 'view' : view,
#                                 'view_accuracy': view_acc,
#                                 'detect_accuray': k[1],
#                                 's3_path':  s3_path
#                                 }
#             else:
#                 print("No Knee Found ... ", i)
#                 result = {'study_id': j[0],
#                                 'dicom_path': path,
#                                 'is_inverted': invert_res,
#                                 'view' : 'NA',
#                                 'view_accuracy': -1,
#                                 'detect_accuray': -1,
#                                 's3_path':  'NA'
#                                 }

#             query = """INSERT INTO knee_ai.preprocessing_pipeline_results ({0}) VALUES {1}""" \
#                     .format(','.join(result.keys()), tuple( [i for i in result.values()] ) )

#             cur.execute(query)
#             conn.commit()

#         except Exception as e:
#             print(e)

#     cur.close()
#     conn.close()


# if __name__ == '__main__':
#     main()