import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np
from keras.models import load_model
import tensorflow_hub as hub
from PIL import Image
from collections import defaultdict
from main import classify_view, invert, crop_knee, classify_knee
import streamlit as st
import time
import altair as alt

st.set_page_config(page_title='5C Network', page_icon = 'https://static.5cnetwork.com/images/5c-new-logo.svg' , layout = 'centered', initial_sidebar_state = 'auto')

st.title("OsteoCheck")

content = st.file_uploader("Upload your scan ", type=["jpg", "jpeg", "png"])


if content is not None:
    with st.spinner("Osteocheck in Progress"):
        time.sleep(2)

        content = Image.open(content)

        st.image(content, width = 300) 


    

        img, invert_res = invert(content)

        if invert_res:
            with st.spinner('The given image is negative. So inverting it...'):
                time.sleep(3)
                
                st.text('✅ Inverted')

        
        with st.spinner('Clasifying view...'):
                time.sleep(3)
        view, view_acc = classify_view(img)

        
        if view != 'Lateral':
            st.text('The given image is in AP view')
            image = crop_knee(img)

            if len(image):
                st.image(image[0][0], width = 300, caption = "Key Image") 
                img = image[0][0]
        
            res, res_acc = classify_knee(img)
            if res == 'Abnormal':
                st.text('The given Knee XRay has an osteoarthritis with a probability of {0:.1f}%.'.format(res_acc*100))
            else:
                st.text('The given Knee XRay seems to be normal with a probability of {0:.1f}%.'.format(res_acc*100))

        else:
            st.text("⚠️ \nThe given image is in Lateral view.\nThe model is currently built to identify osteoarthritis only in AP view. \n ")
                    