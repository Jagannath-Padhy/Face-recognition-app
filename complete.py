from face_detector import YoloDetector
import cv2
import os
import numpy as np
from architecture import *
import pickle
from sklearn.preprocessing import Normalizer
import json

# Load the YoloDetector model
model = YoloDetector(target_size=720, device="cpu", min_face=90)

# Paths and variables
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
l2_normalizer = Normalizer('l2')

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

# Folder containing subfolders of images
faces_folder = "faces/"

encoding_dict = {}

# Iterate through the subfolders (riya, shina, etc.)
for person_name in os.listdir(faces_folder):
    person_folder = os.path.join(faces_folder, person_name)
    if os.path.isdir(person_folder):
        # Process images in the person's folder
        encodes = []
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                org_img = cv2.imread(image_path)
                # Convert the image to RGB
                org_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
                
                # Perform face detection and obtain bounding boxes and keypoints
                bboxes, points = model.predict(org_img_rgb)
                
                for bbox in bboxes[0]:
                    x1, y1, x2, y2 = bbox
                    face_img = org_img_rgb[y1:y2, x1:x2]
                    # Resize and preprocess the face image
                    resized_face_img = cv2.resize(face_img, (160, 160))
                    resized_face_img = resized_face_img.astype('float32') / 255.0
                    face_d = resized_face_img.reshape((1, 160, 160, 3))
                    
                    # Encode the face
                    encode = face_encoder.predict(face_d)[0]
                    encodes.append(encode)
        
        # if encodes:
        #     encode = np.sum(encodes, axis=0)
        #     encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        #     encoding_dict[person_name] = encode
        if encodes:
            encode = np.sum(encodes, axis=0)
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            encoding_dict[person_name] = encode.tolist()

# Save encoding_dict to a pickle file
pickle_path = 'encodings/encodings.pkl'
with open(pickle_path, 'wb') as file:
    pickle.dump(encoding_dict, file)
    
# Save encoding_dict to a JSON file
json_path = 'encodings/encodings.json'
with open(json_path, 'w') as json_file:
    json.dump(encoding_dict, json_file, indent=4)

model_path = "trained_model.h5"
face_encoder.save(model_path)