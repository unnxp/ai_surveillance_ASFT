"""Enroll consented local face photos into localhost Central; never uploads images."""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import requests

from common.identity_schemas import FaceRegistryUpsert
from edge.face_pipeline import FaceExtractor, load_face_settings

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--zone", action="append", default=[], metavar="CAMERA:ZONE")
    parser.add_argument("--consent-confirmed", action="store_true")
    parser.add_argument("--central-url", default="http://127.0.0.1:8000")
    parser.add_argument("--face-config", type=Path,
                        default=ROOT / "configs" / "face_runtime.yaml")
    return parser


def validate_args(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.consent_confirmed is not True:
        raise ValueError("Explicit consent is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.person_id):
        raise ValueError("Invalid person ID")
    if not args.name.strip() or not args.role.strip() or not 1 <= len(args.image) <= 3:
        raise ValueError("Invalid enrollment metadata")
    url = urlsplit(args.central_url)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}):
        raise ValueError("Central URL must be a plain localhost HTTP origin")
    grants = []
    for entry in args.zone:
        camera, separator, zone = entry.partition(":")
        if not separator or not camera.strip() or not zone.strip():
            raise ValueError("Zone grants use CAMERA:ZONE")
        grants.append({"camera_id": camera.strip(), "zone_id": zone.strip()})
    if len({(item["camera_id"], item["zone_id"]) for item in grants}) != len(grants):
        raise ValueError("Duplicate zone grant")
    if any(not path.is_file() for path in args.image):
        raise ValueError("All enrollment images must be local files")
    return grants


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        grants = validate_args(args)
        settings = load_face_settings(args.face_config)
        extractor = FaceExtractor(ROOT / "models" / "det_10g.onnx",
                                  ROOT / "models" / "w600k_r50.onnx", settings)
        templates = []
        for path in args.image:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Enrollment image cannot be read")
            sample = extractor.extract_person(
                image, [0, 0, image.shape[1], image.shape[0]], require_single_face=True)
            if sample is None:
                raise ValueError("Enrollment needs exactly one quality-qualified face per image")
            vector, _quality = sample
            templates.append({"model_version": settings.model_version, "vector": vector})
        payload = FaceRegistryUpsert.model_validate({
            "display_name": args.name, "role": args.role, "consent_confirmed": True,
            "allowed_zones": grants, "templates": templates,
        })
        with requests.Session() as session:
            session.trust_env = False
            response = session.put(
                args.central_url.rstrip("/") + "/api/v1/face-registry/" + args.person_id,
                json=payload.model_dump(mode="json"), timeout=5, allow_redirects=False,
            )
        if response.status_code not in {200, 201}:
            raise RuntimeError("Central rejected enrollment")
        print("Local face enrollment accepted.")
        return 0
    except Exception:
        print("Enrollment failed. Check consent, local images, face quality and Central.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
