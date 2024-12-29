import cv2
import numpy as np
from architecture import *
from train import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from keras.models import load_model
import pickle
from face_detector import YoloDetector

confidence_t = 0.99
recognition_t = 0.3
required_size = (160, 160)
yolo_detector = YoloDetector(target_size=720, device="cuda", min_face=90)
def get_face(img, bbox):
    x1, y1, x2, y2 = bbox
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect_yolo(img, yolo_detector, face_encoder, encoding_dict):
    bboxes, _ = yolo_detector.predict(img)

    for bbox in bboxes[0]:
        x1, y1, x2, y2 = bbox
        face, pt_1, pt_2 = get_face(img, (x1, y1, x2, y2))
        encode = get_encode(face_encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'
        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 200, 200), 2)
    return img

if __name__ == "__main__":
    required_shape = (160, 160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    encoding_dict = load_pickle(encodings_path)
    yolo_detector = YoloDetector(target_size=720, device="cuda", min_face=90)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("CAM NOT OPENED")
            break
        frame = detect_yolo(frame, yolo_detector, face_encoder, encoding_dict)
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
