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
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.5,
        imgsz: int = 640,
        device: int | str = 0,
        skip_frames: int = 2,
        half: bool = True,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.skip_frames = skip_frames
        self.half = half

        self._frame_count = 0
        self._last_results = None
        self._last_batch_size = 0   # ✅ จำ batch size ล่าสุดไว้เปรียบเทียบ

    # ────────────────────────────────────────────
    def detect_batch(self, frames: list):
        """
        Batch Inference: ส่ง N frames เข้า GPU ใน 1 call
        ✅ Fix: ถ้า batch size เปลี่ยน (เช่น cam2 เพิ่งเปิด/ปิด)
                → บังคับ infer ใหม่ทันที ไม่ใช้ cache เดิม
        """
        if not frames:
            return []

        self._frame_count += 1
        should_infer = (self._frame_count % self.skip_frames == 0)

        # ✅ KEY FIX: batch size เปลี่ยน → ต้อง infer ใหม่เสมอ
        batch_size_changed = (len(frames) != self._last_batch_size)
        if batch_size_changed:
            should_infer = True
            self._frame_count = 0   # reset counter ให้ skip นับใหม่

        if should_infer or self._last_results is None:
            results = self.model(
                frames,
                conf=self.conf,
                classes=DETECT_CLASSES,
                device=self.device,
                imgsz=self.imgsz,
                half=self.half,
                verbose=False,
            )
            self._last_results = results
            self._last_batch_size = len(frames)   # ✅ อัปเดต batch size

        return self._last_results

    # ────────────────────────────────────────────
    def detect(self, frame):
        """Single frame detect"""
        results = self.detect_batch([frame])
        return results[0]

    # ────────────────────────────────────────────
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

    def reset(self):
        self._frame_count = 0
        self._last_results = None
        self._last_batch_size = 0
