import cv2
from glob import glob
import os
import random
import torch
import easyocr
import string
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model = YOLO('yolov8m.pt')
pretrained_model.model.to(device)

dataset = './datasets/data.yaml'
backbone = YOLO("yolov8n.pt")
backbone.model.to(device)
results = backbone.train(data = dataset, epochs = 20)

results = backbone.val()

results = backbone('./datasets/test/images/xemay207_jpg.rf.3f203a276200b60478d8e08afc12c930.jpg')
backbone_small = YOLO("yolov8s.pt")
results_medium = backbone_small.train(data=dataset, epochs=100)

success = backbone.export(imgsz = 640, format = 'torchscript', optimize = False, half = False, int8 = False)

np_model = YOLO('runs/detect/train2/weights/best.torchscript')

video = cv2.VideoCapture("./TeslaCam Footage/front.mp4")
ret, frame = video.read()

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('yolotesttrained.mp4', fourcc, 20.0, size)

coco_model = YOLO('yolov8s.pt')

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}

dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

reader = easyocr.Reader(['en'], gpu=True)

def license_complies_format(text):
    if len(text) != 7:
        return False
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    return False
    
def format_license(text):
    license_plate = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate += mapping[j][text[j]]
        else:
            license_plate += text[j]
    return license_plate

def write_csv(results, output_path):
    header = ('frame_number,track_id,car_bbox,car_bbox_score,license_plate_bbox,'
              'license_plate_bbox_score,license_plate_number,license_text_score\n')
    with open(output_path, 'w') as f:
        f.write(header)
        for frame_number, frame_data in results.items():
            for track_id, track_data in frame_data.items():
                car = track_data.get('car')
                lp = track_data.get('license_plate')
                if car and lp and 'number' in lp:
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_number, track_id,
                        '[{} {} {} {}]'.format(*car['bbox']),
                        car['bbox_score'],
                        '[{} {} {} {}]'.format(*lp['bbox']),
                        lp['bbox_score'], lp['number'], lp['text_score']
                    ))

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

results = {}
video = cv2.VideoCapture(video)

ret, frame_number = True, -1
vehicles = {2, 3, 5}
while ret:
    ret, frame = video.read()
    frame_number += 1
    if not ret:
        break
    results[frame_number] = {}
    detections = coco_model.track(frame, persist=True)[0]
    for x1, y1, x2, y2, track_id, score, class_id in detections.boxes.data.tolist():
        if int(class_id) not in vehicles or score <= 0.5:
            continue
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        license_plates = np_model(roi)[0]
        for plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ in license_plates.boxes.data.tolist():
            plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            _, plate_threshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
            np_text, np_score = read_license_plate(plate_threshold)
            if np_text:
                results[frame_number][track_id] = {
                    'car': {'bbox': [x1, y1, x2, y2], 'bbox_score': score},
                    'license_plate': {
                        'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                        'bbox_score': plate_score,
                        'number': np_text,
                        'text_score': np_score
                    }
                }


write_csv(results, './results.csv')
video.release()