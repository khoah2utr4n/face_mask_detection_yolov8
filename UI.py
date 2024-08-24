import av
import cv2
import os
import streamlit as st
import numpy as np
from utils import draw_bounding_boxes
from model import load_model, get_prediction
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes


st.title("Simple UI for Face Mask Detection")
os.makedirs('camera', exist_ok=True) # Directory to store images

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    cv2.imwrite('camera/realtime.png', img)
    predicted_boxes = get_prediction(model, 'camera/realtime.png')
    predicted_image = draw_bounding_boxes(img, predicted_boxes, with_confidence_score=True, is_rgb=False)
    return av.VideoFrame.from_ndarray(predicted_image, format="bgr24")


# Upload new model's weights
os.makedirs('model_weights_UI', exist_ok=True)
is_upload_weights = st.checkbox('Upload new weights?')
if is_upload_weights:
    uploaded_weights = st.file_uploader("Upload the file of model's weights (.pt)")
    if uploaded_weights is not None:
        with open(f"model_weights_UI/weights.pt", mode='wb') as f:
            f.write(uploaded_weights.getbuffer())

pretrained_weights = 'model_weights_UI/weights.pt'
if not os.path.exists(pretrained_weights):
    st.error('Please upload an weights to continue!')
else:
    model = load_model(pretrained_weights)
    st.success('Success')
    on = st.toggle("Using real-time camera")

    if on:
        webrtc_streamer(
            key="example", 
            video_frame_callback=video_frame_callback,
            video_html_attrs=VideoHTMLAttributes(
            autoPlay=True, controls=True, style={"width": "100%"}, muted=True),
        )
        
    else:
        uploaded_file = st.file_uploader('Upload image to predict')
        cols = st.columns(2)

        if uploaded_file is not None:
            # Read the image and reshape 
            uploaded_image = Image.open(uploaded_file).convert('RGB')
            resize_shape = (640, 640 * uploaded_image.size[1] // uploaded_image.size[0])
            uploaded_image = uploaded_image.resize(resize_shape)
            uploaded_image.save('camera/uploaded_image.png')
            uploaded_image = np.array(uploaded_image)
            
            # Get the prediction bounding box and draw on image
            predicted_boxes = get_prediction(model, 'camera/uploaded_image.png')
            predicted_image = draw_bounding_boxes(uploaded_image, predicted_boxes, with_confidence_score=True)
            
            cols[0].image(uploaded_image, caption="Original image")
            cols[1].image(predicted_image, caption="Predicted image")