"""Face extraction and scheduling tests use invented frames and fake inference only."""

import numpy as np
import pytest

from tests.test_face_matching import unit_vector


def frame():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[10:90, 10:90] = np.indices((80, 80)).sum(axis=0)[:, :, None] % 2 * 255
    return image


def test_extractor_rejects_missing_or_poor_quality_face_without_embedding(tmp_path):
    from edge.face_pipeline import FaceExtractor, FaceRuntimeSettings

    class Detector:
        def __init__(self, face):
            self.face = face

        def detect(self, crop):
            return [] if self.face is None else [self.face]

    class Embedder:
        def get_embedding(self, aligned):
            return np.array(unit_vector(), dtype=np.float32)

    image = frame()
    settings = FaceRuntimeSettings(min_face_size=30, min_sharpness=10)
    missing = FaceExtractor(tmp_path / "det.onnx", tmp_path / "emb.onnx", settings,
                            detector=Detector(None), embedder=Embedder())
    assert missing.extract_person(image, [10, 10, 90, 90]) is None
    tiny = {"bbox": [1, 1, 8, 8, 0.99], "kps": [[2, 2]] * 5, "score": 0.99}
    extractor = FaceExtractor(tmp_path / "det.onnx", tmp_path / "emb.onnx", settings,
                              detector=Detector(tiny), embedder=Embedder())
    assert extractor.extract_person(image, [10, 10, 90, 90]) is None


def test_extractor_clips_bbox_and_returns_normalized_embedding(tmp_path):
    from edge.face_pipeline import FaceExtractor, FaceRuntimeSettings

    class Detector:
        def detect(self, crop):
            assert crop.shape[:2] == (80, 80)
            return [{"bbox": [10, 10, 70, 70, 0.9],
                     "kps": [[25, 30], [55, 30], [40, 45], [30, 60], [50, 60]],
                     "score": 0.9}]

    class Embedder:
        def get_embedding(self, aligned):
            return np.array([2.0] + [0.0] * 511, dtype=np.float32)

    extractor = FaceExtractor(tmp_path / "det.onnx", tmp_path / "emb.onnx",
                              FaceRuntimeSettings(min_face_size=30, min_sharpness=10),
                              detector=Detector(), embedder=Embedder())
    vector, quality = extractor.extract_person(frame(), [-10, -10, 80, 80])
    assert len(vector) == 512
    assert vector[0] == pytest.approx(1.0)
    assert quality == pytest.approx(0.9)


def test_extractor_keeps_large_enrollment_photo_resolution_until_scrfd(tmp_path):
    from edge.face_pipeline import FaceExtractor, FaceRuntimeSettings

    image = np.zeros((800, 800, 3), dtype=np.uint8)
    pattern = np.indices((100, 100)).sum(axis=0) % 2 * 255
    image[200:300, 200:300, :] = pattern[:, :, None]

    class Detector:
        def detect(self, crop):
            assert crop.shape[:2] == (800, 800)
            return [{"bbox": [200, 200, 300, 300, 0.95],
                     "kps": [[225, 230], [275, 230], [250, 250], [230, 275], [270, 275]],
                     "score": 0.95}]

    class Embedder:
        def get_embedding(self, aligned):
            return np.array(unit_vector(), dtype=np.float32)

    extractor = FaceExtractor(tmp_path / "det.onnx", tmp_path / "emb.onnx",
                              FaceRuntimeSettings(), detector=Detector(), embedder=Embedder())
    assert extractor.extract_person(image, [0, 0, 800, 800],
                                    require_single_face=True) is not None


def test_extractor_abstains_from_degenerate_landmarks_instead_of_embedding_body(tmp_path):
    from edge.face_pipeline import FaceExtractor, FaceRuntimeSettings

    class Detector:
        def detect(self, crop):
            return [{"bbox": [10, 10, 70, 70, 0.9],
                     "kps": [[30, 30]] * 5, "score": 0.9}]

    class Embedder:
        def get_embedding(self, aligned):
            return np.array(unit_vector(), dtype=np.float32)

    extractor = FaceExtractor(tmp_path / "det.onnx", tmp_path / "emb.onnx",
                              FaceRuntimeSettings(min_sharpness=10),
                              detector=Detector(), embedder=Embedder())
    assert extractor.extract_person(frame(), [10, 10, 90, 90]) is None


def test_default_extractor_loads_existing_onnx_on_cpu_only(tmp_path, monkeypatch):
    from edge.face_pipeline import FaceExtractor, FaceRuntimeSettings

    detection = tmp_path / "det.onnx"
    embedding = tmp_path / "emb.onnx"
    detection.touch()
    embedding.touch()
    class Session:
        def __init__(self, path, providers):
            assert str(path) in {str(detection), str(embedding)}
            calls.append(providers)

        def get_inputs(self):
            return [type("Input", (), {"name": "input"})()]

        def run(self, outputs, inputs):
            return []

    calls = []
    monkeypatch.setattr("edge.face_pipeline.ort.InferenceSession", Session)
    extractor = FaceExtractor(detection, embedding, FaceRuntimeSettings())
    assert extractor.extract_person(frame(), [10, 10, 90, 90]) is None
    assert calls == [["CPUExecutionProvider"], ["CPUExecutionProvider"]]


def test_pipeline_caps_samples_and_keeps_cameras_isolated():
    from common.identity_schemas import FaceMatchResult
    from edge.face_pipeline import FacePipeline, FaceRuntimeSettings

    class Extractor:
        def extract_person(self, image, bbox):
            return unit_vector(), 0.9

    class Client:
        def match_face(self, request):
            return FaceMatchResult(status="unknown", similarity=0.0,
                                   model_version=request.model_version)

    pipeline = FacePipeline(Extractor(), Client(), FaceRuntimeSettings(
        sample_every_frames=1, max_samples_per_track=2, max_tracks_per_frame=1,
        model_version="synthetic-v1"))
    image = frame()
    pipeline.begin_frame("camera_01")
    first = pipeline.observe("camera_01", 1, image, [10, 10, 90, 90], True, ["storage"])
    blocked = pipeline.observe("camera_01", 2, image, [10, 10, 90, 90], True, ["storage"])
    assert first["identity_sample_id"] == 1
    assert first["identity_confirmed"] is False
    assert blocked["identity_status"] == "unavailable"
    pipeline.begin_frame("camera_01")
    second = pipeline.observe("camera_01", 1, image, [10, 10, 90, 90], True, ["storage"])
    assert second["identity_sample_id"] == 2
    assert second["identity_confirmed"] is True
    pipeline.begin_frame("camera_01")
    assert pipeline.observe("camera_01", 1, image, [10, 10, 90, 90], True,
                            ["storage"])["identity_sample_id"] == 2
    pipeline.begin_frame("camera_02")
    assert pipeline.observe("camera_02", 1, image, [10, 10, 90, 90], True,
                            ["storage"])["identity_sample_id"] == 1


def test_pipeline_abstains_on_central_failure_and_prunes_tracks():
    from edge.face_pipeline import FacePipeline, FaceRuntimeSettings

    class Extractor:
        def extract_person(self, image, bbox):
            return unit_vector(), 0.9

    class Client:
        def match_face(self, request):
            raise RuntimeError("sensitive")

    pipeline = FacePipeline(Extractor(), Client(), FaceRuntimeSettings(
        sample_every_frames=1, model_version="synthetic-v1"))
    pipeline.begin_frame("camera_01")
    result = pipeline.observe("camera_01", 1, frame(), [10, 10, 90, 90], True, ["storage"])
    assert result["identity_status"] == "unavailable"
    assert result["identity_evaluated"] is False
    pipeline.prune("camera_01", set())
    pipeline.begin_frame("camera_01")
    assert pipeline.observe("camera_01", 1, frame(), [10, 10, 90, 90], False,
                            ["storage"])["identity_status"] == "unavailable"


def test_enrollment_requires_explicit_consent_before_any_image_access(tmp_path):
    from tools.enroll_face import build_parser, validate_args

    image = tmp_path / "missing.jpg"
    arguments = build_parser().parse_args([
        "--person-id", "employee_01", "--name", "Example", "--role", "staff",
        "--image", str(image), "--zone", "camera_01:storage",
    ])
    with pytest.raises(ValueError, match="consent"):
        validate_args(arguments)


def test_gateway_rejects_face_in_synthetic_demo_mode_without_model_loading():
    from edge.run_gateway import main

    assert main(["--synthetic", "--demo", "--face"]) == 2


def test_existing_local_onnx_models_accept_synthetic_blank_frame():
    from pathlib import Path

    from edge.face_pipeline import FaceExtractor, FaceRuntimeSettings

    root = Path(__file__).resolve().parents[1]
    detector = root / "models" / "det_10g.onnx"
    embedder = root / "models" / "w600k_r50.onnx"
    if not detector.is_file() or not embedder.is_file():
        pytest.skip("Local face ONNX weights are absent; no download attempted")
    extractor = FaceExtractor(detector, embedder, FaceRuntimeSettings())
    assert extractor.extract_person(np.zeros((128, 128, 3), dtype=np.uint8),
                                    [0, 0, 128, 128]) is None
    embedding = extractor.embedder.get_embedding(np.zeros((112, 112, 3), dtype=np.uint8))
    assert embedding.shape == (512,)
    assert np.all(np.isfinite(embedding))
