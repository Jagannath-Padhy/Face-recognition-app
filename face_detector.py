import joblib
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import copy
import scipy
import pathlib
import warnings
from math import sqrt
from models.common import Conv
from models.yolo import Model
from utils.datasets import letterbox
from utils.preprocess_utils import align_faces
from utils.general import check_img_size, non_max_suppression_face, \
    scale_coords, scale_coords_landmarks, filter_boxes

class YoloDetector:
    def __init__(self, weights_name='yolov5n_state_dict.pt', config_name='yolov5n.yaml', device='cpu', min_face=100, target_size=None, frontal=False):
        self._class_path = pathlib.Path(__file__).parent.absolute()
        self.device = device
        self.target_size = target_size
        self.min_face = min_face
        self.frontal = frontal
        if self.frontal:
            print('Currently unavailable')
        self.detector = self.init_detector(weights_name, config_name)

    def init_detector(self, weights_name, config_name):
        model_path = os.path.join(self._class_path, 'weights/', weights_name)
        config_path = os.path.join(self._class_path, 'models/', config_name)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        detector = Model(cfg=config_path)
        detector.load_state_dict(state_dict)
        detector = detector.to(self.device).float().eval()
        for m in detector.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  
        return detector

    def _preprocess(self,imgs):
        pp_imgs = []
        for img in imgs:
            h0, w0 = img.shape[:2] 
            if self.target_size:
                r = self.target_size / min(h0, w0)
                if r < 1:  
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)

            imgsz = check_img_size(max(img.shape[:2]), s=self.detector.stride.max())  # check img_size
            img = letterbox(img, new_shape=imgsz)[0]
            pp_imgs.append(img)
        pp_imgs = np.array(pp_imgs)
        pp_imgs = pp_imgs.transpose(0, 3, 1, 2)
        pp_imgs = torch.from_numpy(pp_imgs).to(self.device)
        pp_imgs = pp_imgs.float()  
        pp_imgs /= 255.0  
        return pp_imgs
        
    def _postprocess(self, imgs, origimgs, pred, conf_thres, iou_thres):
        bboxes = [[] for i in range(len(origimgs))]
        landmarks = [[] for i in range(len(origimgs))]
        
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        
        for i in range(len(origimgs)):
            img_shape = origimgs[i].shape
            h,w = img_shape[:2]
            gn = torch.tensor(img_shape)[[1, 0, 1, 0]] 
            gn_lks = torch.tensor(img_shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]] 
            det = pred[i].cpu()
            scaled_bboxes = scale_coords(imgs[i].shape[1:], det[:, :4], img_shape).round()
            scaled_cords = scale_coords_landmarks(imgs[i].shape[1:], det[:, 5:15], img_shape).round()

            for j in range(det.size()[0]):
                box = (det[j, :4].view(1, 4) / gn).view(-1).tolist()
                box = list(map(int,[box[0]*w,box[1]*h,box[2]*w,box[3]*h]))
                if box[3] - box[1] < self.min_face:
                    continue
                lm = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                lm = list(map(int,[i*w if j%2==0 else i*h for j,i in enumerate(lm)]))
                lm = [lm[i:i+2] for i in range(0,len(lm),2)]
                bboxes[i].append(box)
                landmarks[i].append(lm)
        return bboxes, landmarks

    def get_frontal_predict(self, box, points):
        cur_points = points.astype('int')
        x1, y1, x2, y2 = box[0:4]
        w = x2-x1
        h = y2-y1
        diag = sqrt(w**2+h**2)
        dist = scipy.spatial.distance.pdist(cur_points)/diag
        predict = self.anti_profile.predict(dist.reshape(1, -1))[0]
        if predict == 0:
            return True
        else:
            return False
    def align(self, img, points):
        crops = [align_faces(img,landmark=np.array(i)) for i in points]
        return crops

    def predict(self, imgs, conf_thres = 0.3, iou_thres = 0.5):
        one_by_one = False
        if type(imgs) != list:
            images = [imgs]
        else:
            images = imgs
            one_by_one = False
            shapes = {arr.shape for arr in images}
            if len(shapes) != 1:
                one_by_one = True
                warnings.warn(f"Can't use batch predict due to different shapes of input images. Using one by one strategy.")

        origimgs = copy.deepcopy(images)

        if one_by_one:
            images = [self._preprocess([img]) for img in images]
            bboxes = [[] for i in range(len(origimgs))]
            points = [[] for i in range(len(origimgs))]
            for num, img in enumerate(images):
                with torch.inference_mode(): 
                    print(single_pred.shape)
                bb, pt = self._postprocess(img, [origimgs[num]], single_pred, conf_thres, iou_thres)
                #print(bb)
                bboxes[num] = bb[0]
                points[num] = pt[0]
        else:
            images = self._preprocess(images)
            with torch.inference_mode(): 
                pred = self.detector(images)[0]
            bboxes, points = self._postprocess(images, origimgs, pred, conf_thres, iou_thres)           
        print("Detected Bounding Boxes:")
        for bbox in bboxes:
            print(bbox)
        return bboxes, points

    def __call__(self,*args):
        return self.predict(*args)

if __name__ == '__main__':
    a = YoloDetector()
