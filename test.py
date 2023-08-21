import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

# Khởi tạo pyttsx3 để phát ra thông báo âm thanh
engine = pyttsx3.init()

# Thiết lập camera (0 cho webcam tích hợp sẵn, 1 cho camera ngoài)
cap = cv2.VideoCapture(0)

# Sử dụng dlib để phát hiện khuôn mặt
face_detector = dlib.get_frontal_face_detector()

# Tải mô hình landmarks cho khuôn mặt
dlib_facelandmark = dlib.shape_predictor("C:\\Users\\ACER\\Downloads\\shape_predictor_68_face_landmarks.dat")

# Hàm để tính tỉ lệ khung hình của mắt
def calculate_aspect_ratio(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_eye

# Cài đặt thời gian chờ giữa các khung hình để giảm tải xử lý
frame_skip_interval = 5
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

        # Xác định điểm cho mắt trái và mắt phải
        eye_points = [(range(42, 48), (0, 255, 0)), (range(36, 42), (255, 255, 0))]

        for point_range, color in eye_points:
            eye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in point_range]

            for n in range(len(eye)):
                x, y = eye[n]
                next_point = n + 1 if n < len(eye) - 1 else 0
                x2, y2 = eye[next_point]
                cv2.line(frame, (x, y), (x2, y2), color, 1)

            if point_range == range(42, 48):
                right_eye = eye
            else:
                left_eye = eye

        # Tính tỉ lệ khung hình cho mắt trái và mắt phải
        right_eye_aspect_ratio = calculate_aspect_ratio(right_eye)
        left_eye_aspect_ratio = calculate_aspect_ratio(left_eye)
        average_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

        # Làm tròn tỉ lệ trung bình của mắt trái và phải
        average_eye_aspect_ratio = round(average_eye_aspect_ratio, 2)

        # Phát hiện tình trạng buồn ngủ dựa trên tỉ lệ khung hình của mắt
        if average_eye_aspect_ratio < 0.25:
            cv2.putText(frame, "DROWSINESS DETECTED", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
            cv2.putText(frame, "Alert!!!! WAKE UP", (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

            # Gọi hàm chuyển văn bản thành giọng để cảnh báo người dùng
            engine.say("Alert!!!! WAKE UP")
            engine.runAndWait()

    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Nhấn 'Esc' để thoát
        break

cap.release()
cv2.destroyAllWindows()
