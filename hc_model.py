import cv2
import numpy as np
import face_recognition
import os

def face_recognition_realtime(threshold=0.6, min_confidence=0.7, min_face_size=60):
    """
    Performs real-time face recognition using the webcam with improved robustness.

    Args:
        threshold (float): The maximum face distance to consider a match valid.  Adjust this based on your dataset and environment.
        min_confidence (float): Minimum confidence for a face detection to be considered valid.  Requires OpenCV DNN face detector. (Not used with Haar cascade here)
        min_face_size (int): Minimum face size in pixels to consider a detection valid.  Helps filter out small, false positives.
    """

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
        """
        Calculates face encodings for a list of images.

        Args:
            images: A list of images (NumPy arrays) containing faces.

        Returns:
            A list of face encodings (NumPy arrays), one for each image.
        """
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

    # Load OpenCV Haar Cascade face detector (more robust)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Use Haar Cascade

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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size))  # Modified minSize

        #imgS = cv2.resize(img, (0,0), None, 0.25,0.25) # Resize for faster processing  (optional)
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Use original image
        #facesCurrentFrame = face_recognition.face_locations(imgS)  # Removed redundant call
        face_locations = []
        for (x, y, w, h) in faces:
            # Check if the face is large enough.  This is the key change.
            if w >= min_face_size and h >= min_face_size:
                face_locations.append((y, x + w, y + h, x)) # Convert to face_recognition format

        # Only encode faces if face_locations is not empty
        if face_locations:
            encodingsCurrentFrame = face_recognition.face_encodings(imgS, face_locations)

            for encodeFace, faceLoc in zip(encodingsCurrentFrame, face_locations):
                matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if faceDis[matchIndex] < threshold:
                    name = classNames[matchIndex].upper()
                    y1,x2,y2,x1 = faceLoc
                    #y1, x2, y2, x1 = (y1*4),(x2*4),(y2*4),(x1*4)  # No need to scale since we didn't resize
                    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),2)
                    cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                else:
                    name = "Unknown"
                    y1,x2,y2,x1 = faceLoc
                    #y1, x2, y2, x1 = (y1*4),(x2*4),(y2*4),(x1*4) # No need to scale
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),2)
                    cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)


        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Example Usage
face_recognition_realtime(threshold=0.55, min_confidence=0.8, min_face_size=70)  # Adjust these values
