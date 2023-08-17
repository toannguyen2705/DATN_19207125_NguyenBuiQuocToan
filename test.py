import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

# INITIALIZING pyttsx3 FOR ALERT AUDIO MESSAGES
engine = pyttsx3.init()

# SETTING UP THE CAMERA (0 for built-in webcam, 1 for external camera)
cap = cv2.VideoCapture(0)

# FACE DETECTION USING dlib
face_detector = dlib.get_frontal_face_detector()

# LOAD THE LANDMARKS MODEL FOR FACE
dlib_facelandmark = dlib.shape_predictor("C:\\Users\\ACER\\Downloads\\shape_predictor_68_face_landmarks.dat")

# FUNCTION TO CALCULATE THE ASPECT RATIO OF THE EYE
def calculate_aspect_ratio(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_eye

# MAIN LOOP TO CONTINUOUSLY PROCESS FRAMES
frame_skip_interval = 5  # Process every 5th frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip_interval != 0:
        continue

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        left_eye = []
        right_eye = []

        # POINTS ALLOCATION FOR THE LEFT EYE (42 to 47 in the .dat file)
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            right_eye.append((x, y))
            next_point = n + 1 if n < 47 else 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        # POINTS ALLOCATION FOR THE RIGHT EYE (36 to 41 in the .dat file)
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            left_eye.append((x, y))
            next_point = n + 1 if n < 41 else 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        # CALCULATING THE ASPECT RATIO FOR LEFT AND RIGHT EYES
        right_eye_aspect_ratio = calculate_aspect_ratio(right_eye)
        left_eye_aspect_ratio = calculate_aspect_ratio(left_eye)
        average_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

        # ROUNDING OFF THE AVERAGE MEAN OF RIGHT AND LEFT EYES
        average_eye_aspect_ratio = round(average_eye_aspect_ratio, 2)

        # DETECTING DROWSINESS BASED ON THE EYE ASPECT RATIO
        if average_eye_aspect_ratio < 0.25:
            cv2.putText(frame, "DROWSINESS DETECTED", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
            cv2.putText(frame, "Alert!!!! WAKE UP", (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

            # CALLING THE TEXT-TO-SPEECH FUNCTION TO ALERT THE PERSON
            engine.say("Alert!!!! WAKE UP")
            engine.runAndWait()

    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()