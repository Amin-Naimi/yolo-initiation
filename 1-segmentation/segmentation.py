from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO("./yolo/yolov8n.pt") # Load a pretrained model

    results = model(source=0, show=True, save=True)