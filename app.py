from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import numpy as np
from architecture import InceptionResNetV2  # Assuming InceptionResNetV2 is defined in architecture module
from training_images import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from keras.models import load_model
import pickle
from face_detector import YoloDetector
import base64
from typing import List
from pathlib import Path
import traceback

UPLOAD_FOLDER = "faces"

app = FastAPI()

confidence_t = 0.99
recognition_t = 0.3
required_size = (160, 160)

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

    # Encode the image to base64
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
    return img_base64

async def process_frame(frame_bytes: bytes):
    try:
        img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img_base64 = detect_yolo(img, yolo_detector, face_encoder, encoding_dict)
            return img_base64.encode('utf-8')
        else:
            print("Error: Decoded image is None")
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing frame: {e}")
        return b""

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            processed_frame = await process_frame(frame_bytes)
            if processed_frame:
                await websocket.send_bytes(processed_frame)
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()

@app.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    faces_folder = Path(UPLOAD_FOLDER)
    faces_folder.mkdir(parents=True, exist_ok=True)

    person_folder = faces_folder / name
    person_folder.mkdir(exist_ok=True)

    image_encodings = []

    for file in files:
        content = await file.read()
        image_path = person_folder / file.filename
        with open(image_path, "wb") as image_file:
            image_file.write(content)

    exec(open("train.py").read())
    return {"""message": "Images enrolled successfully \n "message": "Training completed successfully"""}
    return {}

required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path_m = "facenet_keras_weights.h5"
face_encoder.load_weights(path_m)
encodings_path = 'encodings/encodings.pkl'
encoding_dict = load_pickle(encodings_path)
yolo_detector = YoloDetector(target_size=720, device="cpu", min_face=90)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=True)
