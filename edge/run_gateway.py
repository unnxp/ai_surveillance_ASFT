"""Run finite file playback or an explicit synthetic metadata demo against Central."""

import argparse
import math
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from cameras.video_source import SyntheticSource, VideoFileSource
from clients.central_client import CentralClient, RequestsTransport
from common.schemas import CandidateDraft
from edge.shared_gateway import CameraRuntime, SharedGateway

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResourceSettings:
    max_fps: float
    device: str | int
    half: bool


def load_resources(path: Path) -> ResourceSettings:
    import yaml

    with path.open(encoding="utf-8") as file:
        document = yaml.safe_load(file)
    values = document["resource_limits"]
    fps = float(values["max_fps_per_camera"])
    device = values["yolo_device"]
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid FPS limit")
    half = bool(values["yolo_half"]) and str(device).lower() != "cpu"
    return ResourceSettings(fps, device, half)


class DemoProcessor:
    """Emit one clearly marked test candidate per camera; not a learned prediction."""

    def __init__(self):
        self._emitted: set[str] = set()

    def process(self, camera_id, frame, state):
        if camera_id in self._emitted:
            return ()
        self._emitted.add(camera_id)
        return (CandidateDraft(
            candidate_type="possible_loitering", reason_codes=["synthetic_test_only"],
            attributes={"synthetic": True, "purpose": "pipeline_smoke_test"},
        ),)


class VideoPreview:
    """Live operator window; rendering and the zone editor live in edge.preview_ui."""

    def __init__(self, spatial_engine=None, config_path: Path | None = None,
                 action_lookup=None):
        from edge.preview_ui import PreviewWindow

        self._window = PreviewWindow(spatial_engine, config_path, action_lookup)

    @property
    def should_stop(self) -> bool:
        return self._window.should_stop

    def update(self, frames_by_camera: dict[str, Any], states_by_camera: dict[str, dict]) -> None:
        self._window.update(frames_by_camera, states_by_camera)

    def close(self) -> None:
        self._window.close()


class BlankSource:
    """Infinite empty black video feed for unused/empty camera slots."""
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.finished = False

    def read(self) -> tuple[bool, Any]:
        if self.finished:
            return False, None
        import numpy as np
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        self.finished = True


@dataclass(frozen=True)
class PlaybackSummary:
    cycles: int
    events_sent: int
    heartbeats_sent: int
    failures: int


def run_gateway(
    gateway: SharedGateway, *, fps: float, max_cycles: int = 0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> PlaybackSummary:
    if not math.isfinite(fps) or fps <= 0 or max_cycles < 0:
        raise ValueError("FPS must be finite and positive; max_cycles must be nonnegative")
    cycles = events = heartbeats = failures = 0
    try:
        while True:
            started = monotonic()
            if getattr(gateway, "preview_stop_requested", False):
                break
            result = gateway.run_once()
            if getattr(gateway, "preview_stop_requested", False):
                break
            cycles += 1
            events += result.events_sent
            heartbeats += result.heartbeats_sent
            failures += len(result.failures)
            if max_cycles and cycles >= max_cycles:
                break
            file_sources = [c.source for c in gateway.cameras if isinstance(c.source, VideoFileSource)]
            if file_sources and all(getattr(src, "finished", False) for src in file_sources):
                break
            if all(getattr(camera.source, "finished", False) for camera in gateway.cameras):
                break
            delay = 1 / fps - (monotonic() - started)
            if delay > 0:
                sleep(delay)
    finally:
        for camera in gateway.cameras:
            try:
                camera.source.close()
            except Exception:
                failures += 1
        report = getattr(gateway, "report_stopped", None)
        if report is not None:
            result = report()
            heartbeats += result.heartbeats_sent
            failures += len(result.failures)
    return PlaybackSummary(cycles, events, heartbeats, failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="append", default=[], metavar="ID=FILE")
    parser.add_argument("--synthetic", action="store_true", help="Use two finite synthetic sources")
    parser.add_argument("--demo", action="store_true", help="Generate explicitly marked smoke-test candidates")
    parser.add_argument("--central-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gateway-id", default="site-gateway-01")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--action", action="store_true", help="Enable the local action checkpoint")
    parser.add_argument("--face", action="store_true", help="Enable experimental local Face-Zone candidates")
    parser.add_argument("--face-config", type=Path,
                        default=ROOT / "configs" / "face_runtime.yaml")
    parser.add_argument("--action-weights", type=Path,
                        default=ROOT / "models" / "action_classifier_3cls.pth")
    parser.add_argument("--action-meta", type=Path,
                        default=ROOT / "models" / "action_classifier_3cls_meta.yaml")
    parser.add_argument("--action-config", type=Path,
                        default=ROOT / "configs" / "action_runtime.yaml")
    parser.add_argument("--spatial-config", type=Path,
                        default=ROOT / "configs" / "spatial_rules.yaml")
    parser.add_argument("--action-device", default=None, help="cpu or CUDA device index")
    parser.add_argument("--device", default=None, help="cpu or CUDA device index")
    parser.add_argument("--resources", type=Path, default=ROOT / "configs" / "resource_limits.yaml")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means stop at video EOF")
    parser.add_argument("--preview", action="store_true",
                        help="Display live video playback window with detection overlays")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.synthetic and (not args.demo or args.camera):
            raise ValueError("--synthetic requires --demo and cannot be combined with --camera")
        if args.action and args.demo:
            raise ValueError("Action inference is unavailable in the synthetic demo")
        if args.face and args.demo:
            raise ValueError("Face inference is unavailable in the synthetic demo")
        if not args.synthetic and not args.camera:
            raise ValueError("Supply --camera ID=FILE twice, or --synthetic --demo")
        url = urlsplit(args.central_url)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}):
            raise ValueError("Central URL must be a plain localhost HTTP origin")
        settings = load_resources(args.resources)
        if not math.isfinite(args.fps) or not 0 < args.fps <= settings.max_fps or args.max_cycles < 0:
            raise ValueError("FPS must be positive and within resource limits; max-cycles cannot be negative")
        camera_specs = [("camera_01", None), ("camera_02", None)] if args.synthetic else []
        for specification in args.camera:
            name, separator, path = specification.partition("=")
            if not separator or not name.strip() or not path.strip():
                raise ValueError("Each camera must use ID=FILE")
            cleaned_path = path.strip()
            if cleaned_path.lower() in {"none", "empty", "null", "blank"}:
                camera_specs.append((name.strip(), None))
            else:
                camera_specs.append((name.strip(), Path(cleaned_path)))
        if not args.synthetic and len(camera_specs) == 1:
            first_name = camera_specs[0][0]
            second_name = "camera_02" if first_name != "camera_02" else "camera_01"
            camera_specs.append((second_name, None))
        ids = [name for name, _ in camera_specs]
        if len(ids) != 2 or len(set(ids)) != 2 or not args.gateway_id.strip():
            raise ValueError("This playback MVP requires two unique camera IDs and a gateway ID")
        # Validate files before loading models or opening either capture.
        if any(path is not None and not path.is_file() for _, path in camera_specs):
            raise ValueError("Both video sources must be existing local files")
        if args.action and (not args.action_weights.is_file() or not args.action_meta.is_file()):
            raise FileNotFoundError("Local action checkpoint and metadata are required")
        if args.face and (not args.face_config.is_file()
                          or not (ROOT / "models" / "det_10g.onnx").is_file()
                          or not (ROOT / "models" / "w600k_r50.onnx").is_file()):
            raise FileNotFoundError("Local face configuration and ONNX weights are required")
        if not args.demo:
            from edge.spatial import load_spatial_config

            spatial = load_spatial_config(args.spatial_config)
        transport = RequestsTransport(args.central_url)
        central_client = CentralClient(transport)
        if args.demo:
            processor = DemoProcessor()
            versions = {"processor": "synthetic-smoke-test-v1"}
            evidence_encoder = None
            print("DEMO: candidates are synthetic pipeline checks, not AI predictions.")
        else:
            from edge.detection import DetectionProcessor, SharedYOLODetector
            from edge.evidence import EvidenceEncoder

            evidence_encoder = EvidenceEncoder()
            device = settings.device if args.device is None else args.device
            detector = SharedYOLODetector(args.weights, device=device, half=settings.half)
            face = None
            action_engine = None
            if args.face:
                from edge.face_pipeline import FaceExtractor, FacePipeline, load_face_settings

                face_settings = load_face_settings(args.face_config)
                face = FacePipeline(FaceExtractor(
                    ROOT / "models" / "det_10g.onnx",
                    ROOT / "models" / "w600k_r50.onnx", face_settings,
                ), central_client, face_settings)
            detection = DetectionProcessor(detector, ids, frame_rate=args.fps,
                                           spatial=spatial, face=face)
            processor = detection
            versions = {"detector": args.weights.name, "tracker": "bytetrack"}
            if args.face:
                versions["face"] = face_settings.model_version
                print("Face-Zone mode: experimental candidates; match threshold from configs/face_runtime.yaml (open-set sweep, FAR <= 1%).")
            if args.action:
                from edge.action_pipeline import ActionProcessor
                from edge.action_runtime import ActionEngine, ActionModel, load_action_settings

                action_settings = load_action_settings(args.action_config)
                action_device = (settings.device if args.action_device is None
                                 else args.action_device)
                model = ActionModel(args.action_weights, args.action_meta, device=action_device)
                action_engine = ActionEngine(model, tuple(ids), action_settings)
                processor = ActionProcessor(detection, action_engine)
                versions["action"] = args.action_weights.name
                print("Action mode: 32 fresh frames per track; thresholds are unvalidated baselines.")
            else:
                print("Detection mode: person observations only; use --action for action candidates.")
        preview = None
        if args.preview:
            engine = None if args.demo else action_engine

            def _action_lookup(camera_id, track_id):
                try:
                    return engine.last_decision(camera_id, int(track_id)) if engine else None
                except KeyError:
                    return None

            preview = VideoPreview(spatial_engine=spatial if not args.demo else None,
                                   config_path=args.spatial_config, action_lookup=_action_lookup)
        with ExitStack() as resources:
            resources.callback(transport.close)
            if preview is not None:
                resources.callback(preview.close)
            cameras = []
            for name, path in camera_specs:
                if path is None:
                    source = SyntheticSource() if args.synthetic else BlankSource()
                else:
                    source = VideoFileSource(path)
                resources.callback(source.close)
                cameras.append(CameraRuntime(name, source))

            def _preview_cb(frames, states):
                if preview is not None:
                    preview.update(frames, states)
                    if preview.should_stop:
                        setattr(gateway, "preview_stop_requested", True)

            gateway = SharedGateway(
                cameras, processor, central_client, gateway_id=args.gateway_id,
                gateway_boot_id=str(uuid4()),
                configuration_version=("face-zone-experimental-v1" if args.face else
                                       "week2b-v1" if not args.demo else "week1b-v1"),
                model_versions=versions, clock=lambda: datetime.now(timezone.utc),
                event_id_factory=uuid4,
                evidence_encoder=evidence_encoder,
                preview_callback=_preview_cb if preview is not None else None,
            )
            summary = run_gateway(gateway, fps=args.fps, max_cycles=args.max_cycles)
        print(f"Cycles={summary.cycles} events={summary.events_sent} "
              f"heartbeats={summary.heartbeats_sent} failures={summary.failures}")
        return 1 if summary.failures else 0
    except KeyboardInterrupt:
        print("Gateway stopped.")
        return 130
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, ImportError):
        # Never reproduce raw filenames, URLs, credentials, or library exception messages.
        print("Gateway could not start or complete. Check local files, resources, --demo, "
              "weights and the localhost Central service.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
