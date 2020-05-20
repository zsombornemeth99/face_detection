import cv2

faces=cv2.imread("c:/Users/zsomb/OneDrive/Desktop/faces/faces.png")

gray=cv2.cvtColor(faces,cv2.COLOR_BGR2GRAY)

haar_face=cv2.CascadeClassifier("c:/Users/zsomb/OneDrive/Desktop/faces/haarcascade_frontalface_default.xml")
haar_eye=cv2.CascadeClassifier("c:/Users/zsomb/OneDrive/Desktop/faces/haarcascade_eye.xml")
detected_faces=haar_face.detectMultiScale(gray)

for (x, y, w, h) in detected_faces:
    cv2.rectangle(faces, (x,y), (x+w,y+h), (255,0,0), 2)
    eye_color=faces[y:y+h, x:x+w]
    eye_gray=gray[y:y+h, x:x+w]
    detected_eye=haar_eye.detectMultiScale(eye_gray,1.1,10)
    for (mx, my, mw, mh) in detected_eye:
        cv2.rectangle(eye_color, (mx,my), (mx+mw,my+mh), (0,255,0), 2)


cv2.imshow("Title",faces)
#cv2.imshow("Gray",gray)

cv2.waitKey(0)
cv2.destroyAllWindows()