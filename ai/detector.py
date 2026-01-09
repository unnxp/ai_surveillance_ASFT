from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, classes=[0,24,26,28,43],device=0) 
        #class 0 = person , 24 = backpack, 26 = handbag, 28 = suitcase , 43 = knife
        return results[0]

