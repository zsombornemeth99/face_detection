import cv2

haar_face=cv2.CascadeClassifier("c:/Users/zsomb/OneDrive/Desktop/faces/haarcascade_frontalface_default.xml")
haar_eye=cv2.CascadeClassifier("c:/Users/zsomb/OneDrive/Desktop/faces/haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while(True):
    ret, faces = cap.read()

    gray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
    detected_faces=haar_face.detectMultiScale(gray)

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(faces, (x,y), (x+w,y+h), (255,0,0), 2)
        eye_color=faces[y:y+h, x:x+w]
        eye_gray=gray[y:y+h, x:x+w]
        detected_eye=haar_eye.detectMultiScale(eye_gray,1.1,10)
        for (mx, my, mw, mh) in detected_eye:
            cv2.rectangle(eye_color, (mx,my), (mx+mw,my+mh), (0,255,0), 2)

    cv2.imshow('frame',faces)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()