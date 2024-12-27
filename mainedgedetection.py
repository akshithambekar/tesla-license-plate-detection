import ast
import cv2 as cv
import easyocr
import numpy as np
import pandas as pd
import string
from ultralytics import YOLO

def write_to_csv():
    csv_skeleton = {{}, {}, {}, {}, {}, {}}

coco_model = YOLO('yolov8m.pt')
np_model = YOLO('runs/detect/train5/weights/best.pt')
video = cv.VideoCapture('./Footage/front_trimmed.mp4')

ret, frame = video.read()
frame_width, frame_height = int(video.get(3)), int(video.get(4))
size = (frame_width, frame_height)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('licenseplatedetection.mp4', fourcc, 20.0, size)

ret = True
frame_num = -1
vehicles = [2, 3, 5, 7]
while ret:
    frame_num += 1
    ret, frame = video.read()
    if ret and frame_num < 10:
        detections = coco_model.track(frame, persist = True)[0]
        detections.save_crop('outputs')
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if (int(class_id) in vehicles and score > 0.4):
                vehicle_bb = []
                vehicle_bb.append([x1, y1, x2, y2, track_id, score])
                for bb in vehicle_bb:
                    print(bb)
                    roi = frame[int(y1): int(y2), int(x1): int(x2)]
                    license_plates = np_model(roi)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        px1, py1, px2, py2, p_score, _ = license_plate
                        print(license_plate, 'track_id: ' + str(bb[4]))
                        plate = roi[int(py1): int(py2), int(px1): int(px2)]
                        plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
                        _, plate_threshold = cv.threshold(plate_gray, 200, 210, cv.THRESH_BINARY_INV)
                        cv.imwrite(str(track_id) + '_gray.jpg', plate_gray)
                        cv.imwrite(str(track_id) + '_threshold.jpg', plate_threshold)
out.release()
video.release()