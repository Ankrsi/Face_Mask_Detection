from tensorflow.keras.models import load_model
import numpy as np
import cv2

img_size=224
model=load_model("mask_model.h5")
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vid = cv2.VideoCapture(0)
while (True):
    ret, image = vid.read()
    try:
        resize_img = cv2.resize(image, (img_size, img_size))
        resize_img = np.array(resize_img).reshape(-1,img_size,img_size,3)
        resize_img = resize_img/255.0
        prediction = model.predict(resize_img)
        predindex = np.argmax(prediction, axis=1)
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(grey_img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        if predindex == 1:
            cv2.putText(image, "NO Mask Found", (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        elif predindex == 0:
            cv2.putText(image, "Mask Found", (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
vid.release()
cv2.destroyAllWindows()
