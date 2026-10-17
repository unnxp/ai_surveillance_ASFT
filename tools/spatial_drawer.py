"""Draw one spatial rule from the first frame of a local video, then save YAML."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from cameras.video_source import VideoFileSource
from edge.spatial import validate_spatial_config

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"roi": "inference_rois", "zone": "restricted_zones", "tripwire": "tripwires"}
DIRECTIONS = ("both", "negative_to_positive", "positive_to_negative")


def normalize_points(points: list[tuple[int, int]], width: int, height: int) -> list[list[float]]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if any(not 0 <= x <= width or not 0 <= y <= height for x, y in points):
        raise ValueError("Drawn points must stay within the image")
    return [[x / width, y / height] for x, y in points]


def build_geometry(kind: str, name: str, points: list[tuple[int, int]],
                   width: int, height: int, *, direction: str = "both") -> dict:
    if kind not in KINDS or not isinstance(name, str) or not name.strip():
        raise ValueError("Choose a spatial rule kind and nonblank name")
    normalized = normalize_points(points, width, height)
    if kind == "tripwire":
        if len(normalized) != 2 or direction not in DIRECTIONS:
            raise ValueError("A tripwire needs two points and a valid direction")
        entry = {"id": name.strip(), "start": normalized[0], "end": normalized[1],
                 "direction": direction}
    else:
        if len(normalized) < 3:
            raise ValueError("ROI and zone polygons need at least three points")
        entry = {"id": name.strip(), "polygon": normalized}
    validate_spatial_config({"schema_version": 1, "cameras": {
        "example": {KINDS[kind]: [entry]},
    }})
    return entry


def save_geometry(path: Path, camera_id: str, kind: str, entry: dict) -> None:
    if kind not in KINDS or not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("Choose a camera and spatial rule kind")
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as file:
            document = yaml.safe_load(file)
        validate_spatial_config(document)
    else:
        document = {"schema_version": 1, "cameras": {}}
    camera = document["cameras"].setdefault(camera_id.strip(), {})
    rules = camera.setdefault(KINDS[kind], [])
    if any(existing["id"] == entry.get("id") for existing in rules):
        raise ValueError("A spatial rule with this name already exists")
    rules.append(entry)
    validate_spatial_config(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".spatial-", suffix=".yaml", delete=False) as file:
            temporary = Path(file.name)
            yaml.safe_dump(document, file, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _draw(frame: np.ndarray, kind: str, points: list[tuple[int, int]]) -> np.ndarray:
    preview = frame.copy()
    for point in points:
        cv2.circle(preview, point, 4, (0, 255, 255), -1)
    if len(points) >= 2:
        cv2.polylines(preview, [np.asarray(points, dtype=np.int32)],
                      False, (0, 255, 255), 2)
    cv2.putText(preview, f"{kind}: left click / Backspace undo / Enter save / Esc cancel",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return preview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, metavar="ID=FILE")
    parser.add_argument("--kind", required=True, choices=tuple(KINDS))
    parser.add_argument("--name", required=True)
    parser.add_argument("--direction", choices=DIRECTIONS, default="both")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "spatial_rules.yaml")
    args = parser.parse_args(argv)
    camera_id, separator, video_path = args.camera.partition("=")
    try:
        if not separator or not camera_id.strip() or not video_path.strip():
            raise ValueError("Camera must be ID=FILE")
        with_video = VideoFileSource(video_path)
        try:
            available, frame = with_video.read()
            if not available:
                raise ValueError("Local video has no frame to draw on")
        finally:
            with_video.close()
        points: list[tuple[int, int]] = []
        window = "Spatial rule (local video frame)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            def on_mouse(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN and (args.kind != "tripwire" or len(points) < 2):
                    points.append((x, y))

            cv2.setMouseCallback(window, on_mouse)
            while True:
                cv2.imshow(window, _draw(frame, args.kind, points))
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    return 1
                if key in (8, 127) and points:
                    points.pop()
                if key in (10, 13):
                    entry = build_geometry(args.kind, args.name, points,
                                           frame.shape[1], frame.shape[0],
                                           direction=args.direction)
                    save_geometry(args.config, camera_id, args.kind, entry)
                    print(f"Saved {args.kind} '{entry['id']}' for camera '{camera_id}'.")
                    return 0
        finally:
            cv2.destroyWindow(window)
    except (OSError, RuntimeError, ValueError, cv2.error):
        # Do not print a private video path or camera source from lower-level errors.
        print("Unable to draw or save this local spatial rule.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
