import cv2

class Camera:
    def __init__(self, source, name="Camera"):
        self.cap = cv2.VideoCapture(source)
        self.name = name

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
