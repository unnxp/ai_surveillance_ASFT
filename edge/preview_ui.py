"""Operator preview: readable overlays and an in-window spatial rule editor.

Overlays are drawn after each frame is resized to its display tile, so line
widths and text stay crisp regardless of the source resolution. The editor
writes rules through ``tools.spatial_drawer`` so validation and atomic saves
stay identical to the standalone drawer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# BGR palette
WHITE = (245, 245, 245)
MUTED = (170, 170, 170)
PANEL = (28, 24, 22)
ACCENT = (255, 191, 0)        # toolbar highlight
OK = (96, 200, 72)            # normal track
REGISTERED = (230, 160, 40)   # registered identity
UNKNOWN = (0, 150, 255)       # unknown identity
ALERT = (60, 60, 235)         # restricted zone / alert
ROI = (220, 200, 60)
WIRE = (0, 220, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA
TOOLBAR_H = 46
TILE_W = 800
FOCUS_W = 1280

KIND_LABEL = {"zone": "Restricted Zone", "tripwire": "Tripwire", "roi": "Inference ROI"}
ACTION_ALERTS = {"Property_Crime", "Personal_Violence"}


def _blend(img: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], bg: tuple[int, int, int],
               fg: tuple[int, int, int] = WHITE, scale: float = 0.5, pad: int = 5) -> tuple[int, int]:
    """Text on a solid pill; ``org`` is the pill's bottom-left. Returns pill (w, h)."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    h, w = img.shape[:2]
    x = int(min(max(0, org[0]), max(0, w - tw - 2 * pad)))
    y = int(min(max(th + 2 * pad, org[1]), h))
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), bg, -1, AA)
    cv2.putText(img, text, (x + pad, y - pad), FONT, scale, fg, 1, AA)
    return tw + 2 * pad, th + 2 * pad + base


def _dashed_polygon(img, pts, color, dash=10, thickness=2):
    for i in range(len(pts)):
        p, q = np.array(pts[i], float), np.array(pts[(i + 1) % len(pts)], float)
        length = np.linalg.norm(q - p)
        steps = max(1, int(length // dash))
        for s in range(0, steps, 2):
            a = p + (q - p) * (s / steps)
            b = p + (q - p) * (min(s + 1, steps) / steps)
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, AA)


def _corner_box(img, x1, y1, x2, y2, color, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, AA)
    k = max(6, min(x2 - x1, y2 - y1) // 4)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * k, cy), color, thickness + 1, AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * k), color, thickness + 1, AA)


def _inside(point, polygon) -> bool:
    return cv2.pointPolygonTest(np.asarray(polygon, np.float32), point, False) >= 0


def _track_style(info: dict, action: str | None) -> tuple[tuple[int, int, int], list[str]]:
    status = info.get("identity_status")
    color, parts = OK, []
    if status == "registered":
        color = REGISTERED
        parts.append(str(info.get("display_name") or info.get("person_id") or "Registered"))
    elif status == "unknown":
        color = UNKNOWN
        parts.append("Unknown")
    if info.get("global_person_id"):
        parts.insert(0, str(info["global_person_id"]))
    if action in ACTION_ALERTS:  # "Normal"/"Insufficient" add noise in crowds
        parts.append(action.replace("_", " "))
        color = ALERT
    return color, parts


def render_tile(frame: np.ndarray, camera_id: str, state: dict, rules: dict | None, *,
                width: int = TILE_W, fps: float | None = None, selected: bool = False,
                action_lookup: Callable[[str, int], str | None] | None = None,
                now: float | None = None) -> tuple[np.ndarray, float]:
    """Resize ``frame`` to ``width`` and draw all overlays. Returns (tile, scale)."""
    src_h, src_w = frame.shape[:2]
    scale = width / max(1, src_w)
    tile = cv2.resize(frame, (width, max(1, int(src_h * scale))), interpolation=cv2.INTER_AREA)
    h, w = tile.shape[:2]
    now = time.time() if now is None else now
    rules = rules or {}

    feet = {}
    for track_id, info in state.items():
        bbox = info.get("bbox") if isinstance(info, dict) else None
        if bbox and len(bbox) == 4:
            feet[track_id] = (((bbox[0] + bbox[2]) / 2) / src_w, bbox[3] / src_h)

    def px(points):
        return [(int(x * w), int(y * h)) for x, y in points]

    # Inference ROI: dashed outline
    for roi in rules.get("inference_rois", ()):
        if roi.get("enabled", True) and len(roi.get("polygon", ())) >= 3:
            pts = px(roi["polygon"])
            _dashed_polygon(tile, pts, ROI)
            draw_label(tile, f"ROI  {roi['id']}", (pts[0][0], pts[0][1] - 4), (90, 80, 20), scale=0.45)

    # Restricted zones: translucent fill, pulses while occupied
    for zone in rules.get("restricted_zones", ()):
        if not zone.get("enabled", True) or len(zone.get("polygon", ())) < 3:
            continue
        pts = np.array(px(zone["polygon"]), np.int32)
        occupied = sum(_inside(f, zone["polygon"]) for f in feet.values())
        pulse = 0.5 + 0.5 * np.sin(now * 6) if occupied else 0.0
        overlay = tile.copy()
        cv2.fillPoly(overlay, [pts], ALERT, AA)
        _blend(tile, overlay, 0.22 + 0.18 * pulse)
        cv2.polylines(tile, [pts], True, ALERT, 3 if occupied else 2, AA)
        top = tuple(pts[np.argmin(pts[:, 1])])
        text = f"RESTRICTED  {zone['id']}" + (f"  |  {occupied} inside" if occupied else "")
        draw_label(tile, text, (top[0], top[1] - 6), ALERT, scale=0.48)

    # Tripwires: line, endpoints and crossing direction arrow
    for wire in rules.get("tripwires", ()):
        if not wire.get("enabled", True) or not wire.get("start") or not wire.get("end"):
            continue
        (p1,), (p2,) = px([wire["start"]]), px([wire["end"]])
        cv2.line(tile, p1, p2, (0, 0, 0), 5, AA)
        cv2.line(tile, p1, p2, WIRE, 2, AA)
        for p in (p1, p2):
            cv2.circle(tile, p, 5, WIRE, -1, AA)
        mid = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
        d = np.array([p2[0] - p1[0], p2[1] - p1[1]], float)
        normal = np.array([-d[1], d[0]]) / max(1e-6, np.linalg.norm(d)) * 22
        direction = wire.get("direction", "both")
        if direction in ("both", "negative_to_positive"):
            cv2.arrowedLine(tile, tuple(mid.astype(int)), tuple((mid + normal).astype(int)), WIRE, 2, AA, tipLength=0.4)
        if direction in ("both", "positive_to_negative"):
            cv2.arrowedLine(tile, tuple(mid.astype(int)), tuple((mid - normal).astype(int)), WIRE, 2, AA, tipLength=0.4)
        draw_label(tile, f"TRIPWIRE  {wire['id']}", (p1[0], p1[1] - 8), (0, 120, 150), scale=0.45)

    # People
    for track_id, info in state.items():
        if track_id not in feet:
            continue
        x1, y1, x2, y2 = (int(v * scale) for v in info["bbox"])
        action = action_lookup(camera_id, track_id) if action_lookup else info.get("action_label")
        color, parts = _track_style(info, action)
        if any(_inside(feet[track_id], z["polygon"]) for z in rules.get("restricted_zones", ())
               if z.get("enabled", True) and len(z.get("polygon", ())) >= 3):
            color = ALERT
            parts.append("IN ZONE")
        _corner_box(tile, x1, y1, x2, y2, color)
        compact = (y2 - y1) < 70 and color != ALERT  # small, non-alert boxes: id only
        gid = info.get("global_person_id")
        label = (gid or f"#{track_id}") if compact else "  ".join([f"#{track_id}", *parts])
        draw_label(tile, label, (x1, y1 - 3), color, scale=0.45)
        cv2.circle(tile, ((x1 + x2) // 2, y2), 4, WHITE, -1, AA)
        cv2.circle(tile, ((x1 + x2) // 2, y2), 4, color, 1, AA)

    # HUD
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), PANEL, -1)
    _blend(tile, overlay, 0.78)
    blink = int(now * 2) % 2 == 0
    cv2.circle(tile, (16, 17), 6, ALERT if blink else (40, 40, 140), -1, AA)
    cv2.putText(tile, "LIVE", (28, 22), FONT, 0.5, WHITE, 1, AA)
    cv2.putText(tile, camera_id.upper(), (80, 23), FONT, 0.62, WHITE, 2, AA)
    right = f"{len(feet)} persons   {fps:.1f} FPS   " if fps else f"{len(feet)} persons   "
    right += datetime.fromtimestamp(now).strftime("%H:%M:%S")
    (rw, _), _ = cv2.getTextSize(right, FONT, 0.5, 1)
    cv2.putText(tile, right, (w - rw - 12, 22), FONT, 0.5, MUTED, 1, AA)

    # Legend
    items = [("Normal", OK), ("Registered", REGISTERED), ("Unknown", UNKNOWN), ("Alert / In zone", ALERT)]
    lx, ly = 10, h - 12
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, h - 30), (min(w, 470), h), PANEL, -1)
    _blend(tile, overlay, 0.7)
    for name, c in items:
        cv2.rectangle(tile, (lx, ly - 10), (lx + 12, ly + 1), c, -1, AA)
        cv2.putText(tile, name, (lx + 18, ly), FONT, 0.42, WHITE, 1, AA)
        lx += 22 + cv2.getTextSize(name, FONT, 0.42, 1)[0][0] + 16

    if selected:
        cv2.rectangle(tile, (1, 1), (w - 2, h - 2), ACCENT, 2, AA)
    return tile, scale


# ---------------------------------------------------------------- editor

@dataclass
class Button:
    action: str
    label: str
    key: str
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)


def _icon(img, action, x, y, color):
    """Small line icons drawn in a 18x18 box at (x, y)."""
    if action == "zone":  # pen
        cv2.line(img, (x + 3, y + 15), (x + 14, y + 4), color, 3, AA)
        cv2.line(img, (x + 2, y + 16), (x + 4, y + 14), color, 2, AA)
    elif action == "tripwire":
        cv2.line(img, (x + 2, y + 15), (x + 16, y + 3), color, 2, AA)
        cv2.circle(img, (x + 2, y + 15), 2, color, -1, AA)
        cv2.circle(img, (x + 16, y + 3), 2, color, -1, AA)
    elif action == "roi":
        _dashed_polygon(img, [(x + 2, y + 2), (x + 16, y + 2), (x + 16, y + 16), (x + 2, y + 16)], color, 4, 1)
    elif action == "undo":
        cv2.ellipse(img, (x + 10, y + 10), (6, 6), 0, 180, 450, color, 2, AA)
        cv2.line(img, (x + 4, y + 10), (x + 2, y + 6), color, 2, AA)
        cv2.line(img, (x + 4, y + 10), (x + 8, y + 8), color, 2, AA)
    elif action == "save":
        cv2.line(img, (x + 2, y + 10), (x + 7, y + 15), color, 2, AA)
        cv2.line(img, (x + 7, y + 15), (x + 16, y + 3), color, 2, AA)
    elif action == "cancel":
        cv2.line(img, (x + 3, y + 3), (x + 15, y + 15), color, 2, AA)
        cv2.line(img, (x + 15, y + 3), (x + 3, y + 15), color, 2, AA)


@dataclass
class SpatialEditor:
    """Pure editing state; independent of OpenCV windows so it can be unit tested."""

    config_path: Path | None
    on_saved: Callable[[], None] | None = None
    mode: str | None = None
    camera_id: str | None = None
    points: list[tuple[float, float]] = field(default_factory=list)  # normalized
    cursor: tuple[float, float] | None = None
    toast: tuple[str, tuple[int, int, int], float] | None = None

    def notify(self, text: str, color=OK, seconds: float = 2.5) -> None:
        self.toast = (text, color, time.time() + seconds)

    def start(self, mode: str, camera_id: str | None) -> None:
        if self.config_path is None:
            self.notify("Editing disabled: no spatial config", ALERT)
            return
        if camera_id is None:
            self.notify("Select a camera first", ALERT)
            return
        self.mode, self.camera_id, self.points = mode, camera_id, []
        hint = "click 2 points" if mode == "tripwire" else "click points, right-click / Enter to save"
        self.notify(f"Drawing {KIND_LABEL[mode]} on {camera_id}: {hint}", ACCENT, 4)

    def cancel(self) -> None:
        if self.mode:
            self.notify("Drawing cancelled", MUTED)
        self.mode, self.points = None, []

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def add_point(self, point: tuple[float, float]) -> None:
        if not self.mode:
            return
        self.points.append((min(1.0, max(0.0, point[0])), min(1.0, max(0.0, point[1]))))
        if self.mode == "tripwire" and len(self.points) == 2:
            self.save()

    def _unique_name(self, document: dict) -> str:
        from tools.spatial_drawer import KINDS

        existing = {r["id"] for r in (document.get("cameras", {}).get(self.camera_id, {})
                                      .get(KINDS[self.mode], []) or [])}
        n = 1
        while f"{self.mode}_{n}" in existing:
            n += 1
        return f"{self.mode}_{n}"

    def save(self) -> bool:
        if not self.mode:
            return False
        import yaml
        from tools.spatial_drawer import build_geometry, save_geometry

        size = 10_000  # build_geometry expects pixels; use a fixed virtual canvas
        pixels = [(round(x * size), round(y * size)) for x, y in self.points]
        try:
            document = {}
            if self.config_path.exists():
                document = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            name = self._unique_name(document)
            entry = build_geometry(self.mode, name, pixels, size, size)
            save_geometry(self.config_path, self.camera_id, self.mode, entry)
        except (ValueError, OSError) as error:
            self.notify(f"Not saved: {error}", ALERT, 4)
            return False
        self.notify(f"Saved {KIND_LABEL[self.mode]} '{name}' on {self.camera_id}", OK, 3)
        self.mode, self.points = None, []
        if self.on_saved is not None:
            self.on_saved()
        return True

    def draw(self, tile: np.ndarray) -> None:
        if not self.mode:
            return
        h, w = tile.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y in self.points]
        live = pts + ([(int(self.cursor[0] * w), int(self.cursor[1] * h))] if self.cursor else [])
        if len(live) >= 2:
            closed = self.mode != "tripwire" and len(live) >= 3
            if self.mode == "zone" and len(live) >= 3:
                overlay = tile.copy()
                cv2.fillPoly(overlay, [np.array(live, np.int32)], ALERT, AA)
                _blend(tile, overlay, 0.25)
            cv2.polylines(tile, [np.array(live, np.int32)], closed, ACCENT, 2, AA)
        for i, p in enumerate(pts):
            cv2.circle(tile, p, 6, ACCENT, -1, AA)
            cv2.circle(tile, p, 6, (0, 0, 0), 1, AA)
            cv2.putText(tile, str(i + 1), (p[0] + 8, p[1] - 6), FONT, 0.45, ACCENT, 1, AA)


class PreviewWindow:
    """Composites camera tiles under a toolbar and routes mouse/keyboard input."""

    def __init__(self, spatial_engine=None, config_path: Path | None = None,
                 action_lookup: Callable[[str, int], str | None] | None = None,
                 window_name: str = "AI Surveillance - Live Preview"):
        self.spatial_engine = spatial_engine
        self.action_lookup = action_lookup
        self.window_name = window_name
        self.should_stop = False
        self.selected: str | None = None
        self.editor = SpatialEditor(config_path if spatial_engine is not None else None,
                                    on_saved=self._reload)
        self.buttons = [Button("zone", "Zone", "Z"), Button("tripwire", "Tripwire", "T"),
                        Button("roi", "ROI", "R"), Button("undo", "Undo", "U"),
                        Button("save", "Save", "Enter"), Button("cancel", "Cancel", "Esc")]
        self._layout: list[tuple[str, int, int, int, int]] = []
        self._window_ready = False
        self._last = None
        self._fps: float | None = None

    def _reload(self) -> None:
        from edge.spatial import load_spatial_config

        if self.spatial_engine is not None and self.editor.config_path is not None:
            self.spatial_engine.reload_rules(load_spatial_config(self.editor.config_path).rules)

    def _rules(self, camera_id: str) -> dict:
        rules = getattr(self.spatial_engine, "rules", None) or {}
        return rules.get(camera_id, {})

    # -- input
    def handle_key(self, key: int) -> None:
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if ch == "q" and not self.editor.mode:
            self.should_stop = True
        elif ch in ("z", "t", "r"):
            self.editor.start({"z": "zone", "t": "tripwire", "r": "roi"}[ch], self.selected)
        elif ch == "u":
            self.editor.undo()
        elif key in (13, 10):
            self.editor.save()
        elif key == 27:
            self.editor.cancel()

    def _press(self, action: str) -> None:
        if action in ("zone", "tripwire", "roi"):
            self.editor.start(action, self.selected)
        elif action == "undo":
            self.editor.undo()
        elif action == "save":
            self.editor.save()
        elif action == "cancel":
            self.editor.cancel()

    def handle_mouse(self, event: int, x: int, y: int) -> None:
        if y < TOOLBAR_H:
            if event == cv2.EVENT_LBUTTONDOWN:
                for b in self.buttons:
                    x1, y1, x2, y2 = b.rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self._press(b.action)
            return
        for cam, x0, y0, w, h in self._layout:
            if x0 <= x < x0 + w and y0 <= y < y0 + h:
                point = ((x - x0) / w, (y - y0) / h)
                if self.editor.mode and cam == self.editor.camera_id:
                    self.editor.cursor = point
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.editor.add_point(point)
                    elif event == cv2.EVENT_RBUTTONDOWN:
                        self.editor.save()
                elif event == cv2.EVENT_LBUTTONDOWN and not self.editor.mode:
                    self.selected = cam
                return

    # -- drawing
    def _toolbar(self, width: int) -> np.ndarray:
        bar = np.full((TOOLBAR_H, width, 3), PANEL, np.uint8)
        cv2.putText(bar, "AI SURVEILLANCE", (14, 29), FONT, 0.62, WHITE, 2, AA)
        x = 200
        for b in self.buttons:
            text = f"{b.label}  [{b.key}]"
            tw = cv2.getTextSize(text, FONT, 0.45, 1)[0][0]
            bw = tw + 42
            active = self.editor.mode == b.action
            enabled = self.editor.config_path is not None or b.action == "cancel"
            fill = (90, 70, 20) if active else (52, 46, 42)
            cv2.rectangle(bar, (x, 8), (x + bw, TOOLBAR_H - 8), fill, -1, AA)
            cv2.rectangle(bar, (x, 8), (x + bw, TOOLBAR_H - 8), ACCENT if active else (80, 72, 66), 1, AA)
            color = ACCENT if active else (WHITE if enabled else (100, 100, 100))
            _icon(bar, b.action, x + 9, 14, color)
            cv2.putText(bar, text, (x + 32, 29), FONT, 0.45, color, 1, AA)
            b.rect = (x, 8, x + bw, TOOLBAR_H - 8)
            x += bw + 8
        status = (f"Editing {self.editor.camera_id}" if self.editor.mode
                  else f"Selected: {self.selected or '-'}   [Q] quit")
        sw = cv2.getTextSize(status, FONT, 0.45, 1)[0][0]
        if x + sw + 20 < width:
            cv2.putText(bar, status, (width - sw - 14, 29), FONT, 0.45, MUTED, 1, AA)
        return bar

    def compose(self, frames_by_camera: dict[str, Any], states_by_camera: dict[str, dict],
                now: float | None = None) -> np.ndarray | None:
        cams = [c for c in sorted(frames_by_camera)
                if isinstance(frames_by_camera[c], np.ndarray) and frames_by_camera[c].ndim == 3]
        if not cams:
            return None
        if self.selected not in cams:
            self.selected = cams[0]
        if self.editor.mode and self.editor.camera_id in cams:
            cams, width = [self.editor.camera_id], FOCUS_W
        else:
            width = TILE_W
        tiles = []
        for cam in cams:
            tile, _ = render_tile(frames_by_camera[cam], cam, states_by_camera.get(cam, {}),
                                  self._rules(cam), width=width, fps=self._fps,
                                  selected=(cam == self.selected and len(cams) > 1),
                                  action_lookup=self.action_lookup, now=now)
            if cam == self.editor.camera_id:
                self.editor.draw(tile)
            tiles.append(tile)
        height = max(t.shape[0] for t in tiles)
        tiles = [t if t.shape[0] == height else
                 cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=PANEL)
                 for t in tiles]
        gap = np.full((height, 4, 3), PANEL, np.uint8)
        row = tiles[0]
        self._layout = [(cams[0], 0, TOOLBAR_H, tiles[0].shape[1], height)]
        for cam, tile in zip(cams[1:], tiles[1:]):
            x0 = row.shape[1] + 4
            row = np.hstack([row, gap, tile])
            self._layout.append((cam, x0, TOOLBAR_H, tile.shape[1], height))
        canvas = np.vstack([self._toolbar(row.shape[1]), row])
        toast = self.editor.toast
        if toast and toast[2] > time.time():
            text, color, _ = toast
            tw = cv2.getTextSize(text, FONT, 0.55, 1)[0][0]
            draw_label(canvas, text, ((canvas.shape[1] - tw) // 2, canvas.shape[0] - 46), color, scale=0.55, pad=9)
        return canvas

    def update(self, frames_by_camera: dict[str, Any], states_by_camera: dict[str, dict]) -> None:
        now = time.time()
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self._fps = inst if self._fps is None else 0.85 * self._fps + 0.15 * inst
        self._last = now
        canvas = self.compose(frames_by_camera, states_by_camera, now)
        if canvas is None:
            return
        if not self._window_ready:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, lambda e, x, y, f, u: self.handle_mouse(e, x, y))
            self._window_ready = True
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1)
        if key != -1:
            self.handle_key(key & 0xFF)

    def close(self) -> None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
