import cv2
import numpy as np
import face_recognition
import os

def face_recognition_realtime():
    path = 'Loaded'
    images = []
    classNames = []
    mylist = os.listdir(path)
    for cl in mylist:
        curIMG = cv2.imread(f'{path}/{cl}')
        images.append(curIMG)
        classNames.append(os.path.splitext(cl)[0])

    print("Dataset Encoding....")

    def findEncodings(images):
        encodelist = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                encode = face_recognition.face_encodings(img)[0] # Handles cases with no faces
                encodelist.append(encode)
            except IndexError:
                print(f"Warning: No face found in image. Skipping.")

        return encodelist


    encodelistKnown = findEncodings(images)
    if not encodelistKnown:
        print("Error: No face encodings found.  Make sure your imagesAttendance folder contains valid face images.")
        return

    print("Encoding Complete")


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return


    cap.set(3, 1280)  # Set width
    cap.set(4, 720)  # Set height

    while True:
        success, img = cap.read()

        if not success:
            print("Error: Could not read frame from webcam.")
            break

        imgS = cv2.resize(img, (0,0), None, 0.25,0.25) # Resize for faster processing
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = face_recognition.face_locations(imgS)
        encodingsCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodingsCurrentFrame, facesCurrentFrame):
            matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            # Add a threshold for face distance to identify unknown faces
            threshold = 0.6  # Adjust this value based on your needs

            if faceDis[matchIndex] < threshold: # Check if the best match is within the threshold.
                name = classNames[matchIndex].upper()
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = (y1*4),(x2*4),(y2*4),(x1*4) # Scale back to original size
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            else:
                name = "Unknown" # label as unknown if no good match
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = (y1*4),(x2*4),(y2*4),(x1*4) # Scale back to original size
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),2) # Red rectangle for unknown faces
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED) # Red box
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)


        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


face_recognition_realtime()
