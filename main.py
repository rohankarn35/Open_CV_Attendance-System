import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
#accessing the camera
video_capture = cv2.VideoCapture(0)

#Saving the face data
my_image = face_recognition.load_image_file("training_face\Avatar.jpg")
my_encoding = face_recognition.face_encodings(my_image)[0]
#saving the known encodings and names in the list
known_encodings = [my_encoding]
know_names = ["Rohan Karn"]
students = know_names.copy()
face_locations = []
face_encodings = []
#storing the current date time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces in the video capture
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings_current = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings_current:
        face_matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if (face_matches[best_match_index]):
            name = know_names[best_match_index]
        if name in know_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomlefttext = (10, 100)
            fontscale = 1.5
            fontcolor = (255, 0, 0)
            thickness = 3
            linetype = 2
            cv2.putText(frame, name + " Present", bottomlefttext, font, fontscale, fontcolor, thickness, linetype)
            if name in students:
                students.remove(name)
                current_date = now.strftime("%H-%M")
                lnwriter.writerow([name, current_date])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()
