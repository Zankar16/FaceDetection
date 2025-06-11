import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)
zankar_image = face_recognition.load_image_file("photo\zankar.png")
zankar_encoding = face_recognition.face_encodings(zankar_image)[0]

khamar_image = face_recognition.load_image_file("photo\khamar.png")
khamar_encoding = face_recognition.face_encodings(khamar_image)[0]

masi_image = face_recognition.load_image_file("photo\masi.png")
masi_encoding = face_recognition.face_encodings(masi_image)[0]

viny_image = face_recognition.load_image_file("photo\Viny.png")
viny_encoding = face_recognition.face_encodings(viny_image)[0]

diya_image = face_recognition.load_image_file("photo\diya.png")
diya_encoding = face_recognition.face_encodings(diya_image)[0]

ananya_image = face_recognition.load_image_file("photo\Ananya.png")
ananya_encoding = face_recognition.face_encodings(ananya_image)[0]

aayushi_image = face_recognition.load_image_file("photo\Aayushi.png")
aayushi_encoding = face_recognition.face_encodings(aayushi_image)[0]

priyanka_image = face_recognition.load_image_file("photo\Priyanka .png")
priyanka_encoding = face_recognition.face_encodings(priyanka_image)[0]

known_face_encodings = [
                         zankar_encoding,
                         masi_encoding,
                         viny_encoding,
                         diya_encoding,
                         ananya_encoding,
                         aayushi_encoding,
                         priyanka_encoding,
                         khamar_encoding,
                        ]
known_face_names = [
                    "zankar",
                    "masi",
                    "vini",
                    "diya",
                    "ananya",
                    "aayushi",
                    "priyanka",
                    "khamar",
                     ] 

while True:

    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      if face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index] :
         name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    cv2.imshow('Face Recognition Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
cap.release()
cv2.destroyAllWindows()



