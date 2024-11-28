# 🎥 Live Gender and Age Detection App  
🔍 **Detect gender and age in live video using OpenCV and Deep Learning models!**  

This project leverages pre-trained models to identify faces in a live webcam feed, predict their gender, and estimate their age. The application is built using **OpenCV** and **Streamlit**, allowing a user-friendly interface to interact with the system.

---

## 📌 Features  

- 🧑‍🤝‍🧑 **Real-time Face Detection** using pre-trained OpenCV DNN models.  
- 👨‍💻 **Gender Prediction**: Distinguishes between male and female faces.  
- 🕰️ **Age Estimation**: Estimates age into predefined ranges.  
- 🎨 **Streamlit Integration**: Interactive web-based interface with dynamic video feed.  
- ⚡ **User-Friendly**: Easy to run and visualize predictions on your own webcam.  

---

## 🛠️ Technologies Used  

- **Programming Language**: Python 🐍  
- **Libraries**:  
  - OpenCV  
  - Streamlit  
  - NumPy  
- **Models**:  
  - OpenCV DNN models for face detection, gender, and age classification.
### Download Pre-trained Models  

Download the following models and place them in the project directory:  

#### 🔎 Face Detection:  
- [Face Detection Config (`opencv_face_detector.pbtxt`)](https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/opencv_face_detector.pbtxt)  
- [Face Detection Model (`opencv_face_detector_uint8.pb`)](https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/opencv_face_detector_uint8.pb)  

#### 👨‍💻 Gender Prediction:  
- [Gender Config (`gender_deploy.prototxt`)](https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_gender_deploy.prototxt)  
- [Gender Model (`gender_net.caffemodel`)](https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_gender.caffemodel)  

#### 🕰️ Age Prediction:  
- [Age Config (`age_deploy.prototxt`)](https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_age_deploy.prototxt)  
- [Age Model (`age_net.caffemodel`)](https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_age.caffemodel)  

## 🧠 Model Information  

### Gender Labels:  
- Male 👨  
- Female 👩  

### Age Labels:  
- `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`  

---

## 🙌 Acknowledgments  

- Models and configurations are sourced from OpenCV’s pre-trained DNN models.  
- Thanks to the developers of **OpenCV** and **Streamlit** for providing powerful tools for building such applications.  

---

## 💬 Contact  

Feel free to reach out for feedback, issues, or contributions!  
📧 **Your Email**: [massnaveen1002@gmail.com](mailto:massnaveen1002@gmail.com)   
