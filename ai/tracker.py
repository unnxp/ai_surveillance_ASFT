import cv2
import numpy as np
import time
from collections import defaultdict, deque


# ══════════════════════════════════════════════════
#  KALMAN FILTER PER TRACK
# ══════════════════════════════════════════════════
class KalmanTrack:
    """
    Kalman Filter ต่อ 1 track_id
    State:       [cx, cy, vx, vy]
    Measurement: [cx, cy]
    """

    def __init__(self):
        # 4 state variables, 2 measurement variables
        self.kf = cv2.KalmanFilter(4, 2)

        # H: measurement matrix
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)

        # F: transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)

        # Q: process noise (ลด = เชื่อ model, เพิ่ม = เชื่อ measurement)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01

        # R: measurement noise (ลด = เชื่อ sensor, เพิ่ม = smooth มากขึ้น)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0

        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        """
        รับ raw measurement → คืน smoothed position
        """
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])

        if not self._initialized:
            # กำหนด initial state จาก measurement แรก
            self.kf.statePre  = np.array([[cx], [cy], [0], [0]], np.float32)
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self._initialized = True

        self.kf.predict()
        corrected = self.kf.correct(meas)
        return float(corrected[0]), float(corrected[1])

    def predict_future(self, n_steps: int = 20) -> list[tuple[float, float]]:
        """
        Predict n_steps ล่วงหน้าจาก state ปัจจุบัน
        คืน list of (x, y) future positions
        """
        if not self._initialized:
            return []

        state = self.kf.statePost.copy()
        future_pts = []
        for _ in range(n_steps):
            state = self.kf.transitionMatrix @ state
            future_pts.append((float(state[0]), float(state[1])))
        return future_pts


# ══════════════════════════════════════════════════
#  TRAJECTORY MANAGER
# ══════════════════════════════════════════════════
class TrajectoryManager:
    """
    จัดการ trajectory ของแต่ละ track_id
    ✅ Kalman smoothing → เส้นไม่หยัก
    ✅ Future prediction → dotted line ข้างหน้า
    """

    _COLOR_CACHE: dict = {}

    def __init__(
        self,
        max_history: int = 60,
        max_age: float = 5.0,
        speed_window: int = 8,
        predict_steps: int = 20,    # predict กี่ frame ล่วงหน้า
    ):
        self.max_history = max_history
        self.max_age = max_age
        self.speed_window = speed_window
        self.predict_steps = predict_steps

        # เก็บ smoothed trajectory per track_id
        self._tracks: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._kalman: dict[int, KalmanTrack] = {}
        self._last_seen: dict[int, float] = {}

    # ══════════════════════════════════════════════
    #  UPDATE
    # ══════════════════════════════════════════════
    def update(self, result, scale: float = 1.0) -> dict[int, float]:
        """
        อัปเดต trajectory จาก YOLO result
        ✅ scale: แปลง YOLO coord → display coord
        ✅ Kalman: smooth ก่อนเก็บ trajectory
        """
        now = time.monotonic()
        speeds: dict[int, float] = {}

        if result is None or result.boxes is None:
            return speeds
        if result.boxes.id is None:
            return speeds

        for box, track_id in zip(
            result.boxes.xyxy,
            result.boxes.id.int().tolist()
        ):
            x1, y1, x2, y2 = box.tolist()
            raw_cx = ((x1 + x2) / 2) * scale
            raw_cy = ((y1 + y2) / 2) * scale

            # ✅ Kalman smooth
            if track_id not in self._kalman:
                self._kalman[track_id] = KalmanTrack()

            smooth_cx, smooth_cy = self._kalman[track_id].update(raw_cx, raw_cy)

            self._tracks[track_id].append((smooth_cx, smooth_cy, now))
            self._last_seen[track_id] = now
            speeds[track_id] = self.get_speed(track_id)

        self._cleanup(now)
        return speeds

    # ══════════════════════════════════════════════
    #  SPEED
    # ══════════════════════════════════════════════
    def get_speed(self, track_id: int) -> float:
        """คืน speed หน่วย pixel/วินาที (จาก smoothed trajectory)"""
        pts = list(self._tracks.get(track_id, []))
        n = min(self.speed_window, len(pts))
        if n < 2:
            return 0.0
        recent = pts[-n:]
        dt = recent[-1][2] - recent[0][2]
        if dt <= 0:
            return 0.0
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return float(np.sqrt(dx ** 2 + dy ** 2) / dt)

    # ══════════════════════════════════════════════
    #  DRAW
    # ══════════════════════════════════════════════
    def draw(self, frame: np.ndarray, result, scale: float = 1.0) -> np.ndarray:
        """
        วาดบน frame:
        1. Trail (smoothed) — เส้นทึบ จาง→สว่าง
        2. Direction arrow
        3. ✅ Prediction — เส้นจุด dotted ล่วงหน้า
        4. Speed + ID label
        """
        if result is None or result.boxes is None:
            return frame
        if result.boxes.id is None:
            return frame

        overlay = frame.copy()

        for box, track_id in zip(
            result.boxes.xyxy,
            result.boxes.id.int().tolist()
        ):
            pts = list(self._tracks.get(track_id, []))
            color = self._get_color(track_id)

            # ── 1. Trail (smoothed) ────────────────
            if len(pts) >= 2:
                points = [(int(p[0]), int(p[1])) for p in pts]
                total = len(points)
                for i in range(1, total):
                    alpha = i / total               # 0=เก่า/จาง, 1=ใหม่/สว่าง
                    thickness = max(1, int(alpha * 4))
                    seg_color = tuple(int(c * alpha) for c in color)
                    cv2.line(overlay, points[i - 1], points[i],
                             seg_color, thickness, cv2.LINE_AA)

            # ── 2. Direction arrow ─────────────────
            if len(pts) >= 4:
                tip  = (int(pts[-1][0]), int(pts[-1][1]))
                tail = (int(pts[-3][0]), int(pts[-3][1]))
                cv2.arrowedLine(overlay, tail, tip, color, 2,
                                tipLength=0.4, line_type=cv2.LINE_AA)

            # ── 3. ✅ Prediction (dotted line) ──────
            if track_id in self._kalman and len(pts) >= 5:
                future_pts = self._kalman[track_id].predict_future(self.predict_steps)
                for i, (fx, fy) in enumerate(future_pts):
                    # วาดทุก 2 step → ดูเป็น dotted
                    if i % 2 == 0:
                        alpha_pred = 1.0 - (i / len(future_pts))  # จางตามระยะ
                        pred_color = tuple(int(c * alpha_pred * 0.8) for c in color)
                        radius = max(1, int(3 * alpha_pred))
                        cv2.circle(overlay, (int(fx), int(fy)),
                                   radius, pred_color, -1, cv2.LINE_AA)

            # ── 4. Speed + ID label ────────────────
            speed = self.get_speed(track_id)
            x1_s = int(box[0] * scale)
            y1_s = int(box[1] * scale)
            label = f"ID:{track_id}  {speed:.0f}px/s"
            cv2.putText(overlay, label, (x1_s, max(0, y1_s - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame

    # ══════════════════════════════════════════════
    #  UTILS
    # ══════════════════════════════════════════════
    def _get_color(self, track_id: int) -> tuple[int, int, int]:
        if track_id not in self._COLOR_CACHE:
            hue = (track_id * 47) % 180
            bgr = cv2.cvtColor(
                np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR
            )[0][0]
            self._COLOR_CACHE[track_id] = (
                int(bgr[0]), int(bgr[1]), int(bgr[2])
            )
        return self._COLOR_CACHE[track_id]

    def _cleanup(self, now: float):
        """ลบ track ที่ไม่ถูก update นานเกิน max_age"""
        stale = [
            tid for tid, t in self._last_seen.items()
            if now - t > self.max_age
        ]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._kalman.pop(tid, None)     # ✅ cleanup Kalman ด้วย
            self._last_seen.pop(tid, None)

    def active_count(self) -> int:
        return len(self._last_seen)

    def reset(self):
        self._tracks.clear()
        self._kalman.clear()                # ✅ reset Kalman ด้วย
        self._last_seen.clear()
