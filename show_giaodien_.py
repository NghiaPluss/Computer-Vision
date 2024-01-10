import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import tkinter as tk
from PIL import ImageTk, Image

#load model
path = r"C:\PYT\Emotion_CV\model\emotion_model.json"
model = model_from_json(open(path, "r").read())

#load weights
path2 = r'C:\PYT\Emotion_CV\model\emotion_model.h5'
model.load_weights(path2)

path3 = r"C:\PYT\Emotion_CV\haarcascades\haarcascade_frontalface_default.xml"
face_haar_cascade = cv2.CascadeClassifier(path3)

def recognize_emotion():
    ret, test_img = cap.read()
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
        roi_gray = gray_img[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = image.resize((640, 480), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
    window.after(15, update_frame)

# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Ứng dụng nhận diện cảm xúc từ khuôn mặt")

# Tạo khung hiển thị videoq
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

# Tạo nút nhận diện cảm xúc
recognize_button = tk.Button(window, text="Nhận diện cảm xúc người trong hình", command=recognize_emotion)
recognize_button.pack(pady=10)

cap = cv2.VideoCapture(0)

update_frame()

window.mainloop()


cap.release()
cv2.destroyAllWindows()
