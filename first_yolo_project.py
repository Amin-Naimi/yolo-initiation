from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Load a model
    model = YOLO("yolov8n.yaml") # build a new model form scratch
    model = YOLO("yolov8n.pt") # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="coco128.yaml", epochs=3) # train the model
    metrics = model.val()
    results = model("https://ultralytics.com/images/bus.jpg", show=True, save=True)
    


