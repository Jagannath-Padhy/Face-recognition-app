from face_detector import YoloDetector
import cv2
import os
import numpy as np
from architecture import *
import pickle
from sklearn.preprocessing import Normalizer
import shutil
import json

model = YoloDetector(target_size=720, device="cpu", min_face=90)

# Paths and variables
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
l2_normalizer = Normalizer('l2')

# Function to normalize an image
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

# Function to get the encoding of a face
def get_encode(face_encoder, face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    encode = face_encoder.predict(face)[0]
    return encode

# Folder paths
faces_folder = "faces/"
faces_done_folder = "faces_done/"
pickle_path = 'encodings/encodings.pkl'

# Load existing encoding dictionary from pickle file if available
if os.path.exists(pickle_path):
    with open(pickle_path, 'rb') as file:
        encoding_dict = pickle.load(file)
else:
    encoding_dict = {}

# Process the images and create the encoding dictionary
for person_name in os.listdir(faces_folder):
    person_folder = os.path.join(faces_folder, person_name)
    if os.path.isdir(person_folder):
        encodes = []
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                org_img = cv2.imread(image_path)
                org_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
                bboxes, _ = model.predict(org_img_rgb)
                for bbox in bboxes[0]:
                    x1, y1, x2, y2 = bbox
                    face_img = org_img_rgb[y1:y2, x1:x2]
                    encode = get_encode(face_encoder, face_img)
                    encodes.append(encode)
        if encodes:
            encode = np.sum(encodes, axis=0)
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            encoding_dict[person_name] = encode.tolist()

            # Move images to the processed folder
            if not os.path.exists(faces_done_folder):
                os.makedirs(faces_done_folder)
            dest_folder = os.path.join(faces_done_folder, person_name)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            for image_name in os.listdir(person_folder):
                shutil.move(os.path.join(person_folder, image_name), os.path.join(dest_folder, image_name))

# Remove empty folders
for root, dirs, files in os.walk(faces_folder):
    for dir in dirs:
        folder = os.path.join(root, dir)
        if not os.listdir(folder): 
            os.rmdir(folder)

# Save encoding_dict to a pickle file
with open(pickle_path, 'wb') as file:
    pickle.dump(encoding_dict, file)
    
# Save encoding_dict to a JSON file
json_path = 'encodings/encodings.json'
with open(json_path, 'w') as json_file:
    json.dump(encoding_dict, json_file, indent=4)