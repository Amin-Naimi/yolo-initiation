import cv2
from ultralytics import YOLO

model = YOLO("./yolo/yolov8n.pt")
results = model("images/2.jpg", show = True, save = True)
cv2.waitKey(0)
