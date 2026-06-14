from ultralytics import YOLO

DETECT_CLASSES = [0, 24, 26, 28, 43]
# 0=person, 24=backpack, 26=handbag, 28=suitcase, 43=knife

DETECT_CLASS_NAMES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    43: "knife",
}


class PersonDetector:
    """
    1 instance = 1 กล้อง = 1 tracker state แยกกันสมบูรณ์

    tracker_mode:
      "bytetrack" → เร็ว ใช้ motion only (อาจ ID สลับถ้าคนเดินสวน)
      "botsort"   → ช้ากว่านิดนึง ใช้ appearance+motion (แยกคนได้ดีกว่า)
    """

    TRACKER_MAP = {
        "bytetrack": "bytetrack.yaml",
        "botsort":   "botsort.yaml",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.5,
        imgsz: int = 640,
        device: int | str = 0,
        half: bool = True,
        name: str = "Detector",
        tracker_mode: str = "botsort",    # ✅ เปลี่ยนตรงนี้
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.name = name

        if tracker_mode not in self.TRACKER_MAP:
            raise ValueError(f"tracker_mode ต้องเป็น {list(self.TRACKER_MAP.keys())}")

        self._tracker_cfg = self.TRACKER_MAP[tracker_mode]
        print(f"[{self.name}] Model: {model_path} | Tracker: {tracker_mode}")

    # ══════════════════════════════════════════════
    #  TRACK  (single frame)
    # ══════════════════════════════════════════════
    def track(self, frame):
        """
        Track single frame พร้อม persist tracker state
        """
        if frame is None:
            return None

        results = self.model.track(
            frame,
            conf=self.conf,
            classes=DETECT_CLASSES,
            device=self.device,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
            persist=True,
            tracker=self._tracker_cfg,
        )
        return results[0]

    # ══════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════
    def has_person(self, result) -> bool:
        if result is None or result.boxes is None:
            return False
        return 0 in result.boxes.cls.tolist()

    def get_class_names(self, result) -> list[str]:
        if result is None or result.boxes is None:
            return []
        return [
            DETECT_CLASS_NAMES.get(int(c), str(int(c)))
            for c in result.boxes.cls.tolist()
        ]
