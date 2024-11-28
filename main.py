import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Paths to model files
faceProto = r"C:\Users\jayas\OneDrive\Desktop\New folder\Gender_and_Age_detector\opencv_face_detector.pbtxt"
faceModel = r"C:\Users\jayas\OneDrive\Desktop\New folder\Gender_and_Age_detector\opencv_face_detector_uint8.pb"
ageProto = r"C:\Users\jayas\OneDrive\Desktop\New folder\Gender_and_Age_detector\age_deploy.prototxt"
ageModel = r"C:\Users\jayas\OneDrive\Desktop\New folder\Gender_and_Age_detector\age_net.caffemodel"
genderProto = r"C:\Users\jayas\OneDrive\Desktop\New folder\Gender_and_Age_detector\gender_deploy.prototxt"
genderModel = r"C:\Users\jayas\OneDrive\Desktop\New folder\Gender_and_Age_detector\gender_net.caffemodel"


facenet = cv2.dnn.readNet(faceModel, faceProto)
agenet = cv2.dnn.readNet(ageModel, ageProto)
gendernet = cv2.dnn.readNet(genderModel, genderProto)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male 👨', 'Female 👩']


def detect_faces_and_attributes(frame):
    frameh, framew = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    facenet.setInput(blob)
    detections = facenet.forward()
    
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  
            x1 = int(detections[0, 0, i, 3] * framew)
            y1 = int(detections[0, 0, i, 4] * frameh)
            x2 = int(detections[0, 0, i, 5] * framew)
            y2 = int(detections[0, 0, i, 6] * frameh)
            
            face = frame[max(0, y1):min(y2, frameh - 1), max(0, x1):min(x2, framew - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            
            gendernet.setInput(blob)
            gender_pred = gendernet.forward()
            gender = genderList[gender_pred[0].argmax()]
            
            
            agenet.setInput(blob)
            age_pred = agenet.forward()
            age = ageList[age_pred[0].argmax()]
            
            results.append((x1, y1, x2, y2, gender, age))
    
    return results


st.title("🎥 Live Gender and Age Detection")
st.markdown("🔍 **Detect gender and age in live video using OpenCV and Deep Learning models!**")


run = st.checkbox("Start Webcam")


if run:
    video = cv2.VideoCapture(0)
    stframe = st.empty()  

    while run:
        ret, frame = video.read()
        if not ret:
            st.error("Unable to access the webcam. Please check your device.")
            break
        
        detections = detect_faces_and_attributes(frame)
        for (x1, y1, x2, y2, gender, age) in detections:
            label = f"{gender}, {age[1:-1]} Years"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    video.release()
else:
    st.info("Click on 'Start Webcam' to begin detection.")
