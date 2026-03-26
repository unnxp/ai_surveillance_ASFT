import cv2
import time
import threading
from collections import deque


class Camera:
    """
    Threaded camera reader with FPS throttle.
    - อ่าน frame ใน background thread
    - ✅ throttle ตาม FPS จริงของวิดีโอ → ไม่เร่ง
    - สำหรับ RTSP/USB กล้องจริง → ไม่ต้อง throttle (กล้องจัดการเอง)
    """

    def __init__(
        self,
        source,
        name: str = "Camera",
        buffer_size: int = 1,
        force_fps: float | None = None,   # บังคับ FPS (None = ใช้ค่าจากไฟล์)
    ):
        self.source = source
        self.name = name
        self.force_fps = force_fps

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ✅ ดึง FPS จากวิดีโอ
        self._native_fps = self._detect_fps()
        self._frame_delay = 1.0 / self._native_fps  # delay ต่อ frame

        # ✅ ตรวจว่าเป็นไฟล์วิดีโอหรือกล้องจริง
        self._is_file = isinstance(source, str) and not source.startswith("rtsp")

        self._buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        print(f"[{self.name}] Source: {source}")
        print(f"[{self.name}] Native FPS: {self._native_fps:.2f} | "
              f"Is File: {self._is_file}")

    # ────────────────────────────────────────────
    def _detect_fps(self) -> float:
        """
        ดึง FPS จากไฟล์วิดีโอ หรือ fallback ถ้าไม่มีข้อมูล
        """
        if self.force_fps:
            return self.force_fps

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        # ✅ กล้อง RTSP/USB มักคืน 0 หรือ ค่าแปลกๆ → fallback 30
        if fps <= 0 or fps > 120:
            fps = 30.0

        return fps

    # ────────────────────────────────────────────
    def start(self):
        """เริ่ม background thread"""
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"Thread-{self.name}",
            daemon=True
        )
        self._thread.start()
        return self

    def _reader_loop(self):
        """
        วนอ่าน frame พร้อม FPS throttle
        - ไฟล์วิดีโอ  → sleep ตาม native FPS
        - กล้องจริง   → อ่านเร็วที่สุด (กล้องจัดการ rate เอง)
        """
        while self._running:
            t_start = time.monotonic()

            ret, frame = self.cap.read()
            if not ret:
                if self._is_file:
                    # วิดีโอจบ → loop กลับต้น
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # กล้องจริงขาดสัญญาณ → รอแล้วลองใหม่
                    time.sleep(0.1)
                    continue

            with self._lock:
                self._buffer.append((ret, frame))

            # ✅ FPS Throttle: เฉพาะไฟล์วิดีโอเท่านั้น
            if self._is_file:
                elapsed = time.monotonic() - t_start
                sleep_time = self._frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # ────────────────────────────────────────────
    def read(self):
        """ดึง frame ล่าสุดจาก buffer (non-blocking)"""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return False, None

    # ────────────────────────────────────────────
    @property
    def native_fps(self) -> float:
        return self._native_fps

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()

    # ────────────────────────────────────────────
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def release(self):
        self.stop()
        self.cap.release()
