import cv2
import face_recognition
import pandas as pd
import os
import datetime

attendance = pd.DataFrame(columns=['ID', 'Name', 'Time'])
known_faces = []
known_names = []
student_images = 'images/'  # Folder where student images are stored

def load_student_images():
    for file_name in os.listdir(student_images):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(student_images, file_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(file_name.split('.')[0])  # Assuming name is the file name

def mark_attendance(name):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    attendance = pd.read_csv('attendance.csv')
    attendance = attendance.append({'ID': name, 'Name': name, 'Time': time}, ignore_index=True)
    attendance.to_csv('attendance.csv', index=False)
    print(f'Attendance marked for {name} at {time}')

def recognize_face():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]  # Convert to RGB (OpenCV uses BGR)
        
        faces = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, faces)

        for (top, right, bottom, left), face_encoding in zip(faces, encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow('Face Recognition Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_student_images()
    recognize_face()
