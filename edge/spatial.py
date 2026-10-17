"""Independent camera-scoped ROI, restricted-zone and tripwire rules."""

from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
from typing import Any

from common.schemas import CandidateDraft


def _point(value: Any) -> tuple[float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(number, bool) or not isinstance(number, (int, float))
                   for number in value)):
        raise ValueError("Spatial points require two normalized numbers")
    x, y = float(value[0]), float(value[1])
    if not all(isfinite(number) and 0 <= number <= 1 for number in (x, y)):
        raise ValueError("Spatial coordinates must be finite and normalized")
    return x, y


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d) -> bool:
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) < 1e-12:
        return False
    delta = (c[0] - a[0], c[1] - a[1])
    t = (delta[0] * s[1] - delta[1] * s[0]) / denominator
    u = (delta[0] * r[1] - delta[1] * r[0]) / denominator
    return 0 <= t <= 1 and 0 <= u <= 1


def _polygon(value: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("Spatial polygon needs at least three vertices")
    vertices = tuple(_point(item) for item in value)
    count = len(vertices)
    area2 = sum(vertices[i][0] * vertices[(i + 1) % count][1]
                - vertices[(i + 1) % count][0] * vertices[i][1]
                for i in range(count))
    if abs(area2) < 1e-8 or any(vertices[i] == vertices[(i + 1) % count]
                                 for i in range(count)):
        raise ValueError("Spatial polygon is degenerate")
    for i in range(count):
        for j in range(i + 1, count):
            if j in {i + 1, (i + count - 1) % count} or (i == 0 and j == count - 1):
                continue
            if _segments_intersect(vertices[i], vertices[(i + 1) % count],
                                   vertices[j], vertices[(j + 1) % count]):
                raise ValueError("Spatial polygon self-intersects")
    return vertices


def _inside(point, polygon) -> bool:
    x, y = point
    contained = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        if (abs(_cross(first, second, point)) < 1e-10
                and min(first[0], second[0]) - 1e-10 <= x <= max(first[0], second[0]) + 1e-10
                and min(first[1], second[1]) - 1e-10 <= y <= max(first[1], second[1]) + 1e-10):
            return True
        if (first[1] > y) != (second[1] > y):
            crossing_x = first[0] + (y - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
            if x < crossing_x:
                contained = not contained
    return contained


def _named(value: Any, allowed: set[str]) -> tuple[str, bool]:
    if not isinstance(value, dict) or not {"id"} <= value.keys() or not value.keys() <= allowed:
        raise ValueError("Invalid spatial entry")
    name = value["id"]
    if (not isinstance(name, str) or not name.strip()
            or not isinstance(value.get("enabled", True), bool)):
        raise ValueError("Spatial entry needs an ID and enabled flag")
    return name.strip(), value.get("enabled", True)


def _count(value: Any, default: int, minimum: int) -> int:
    number = value if value is not None else default
    if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
        raise ValueError("Invalid spatial frame count")
    return number


def _fraction(value: Any, default: float, *, positive: bool = False) -> float:
    number = value if value is not None else default
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ValueError("Invalid spatial threshold")
    number = float(number)
    if not isfinite(number) or not 0 <= number <= 1 or (positive and number == 0):
        raise ValueError("Invalid spatial threshold")
    return number


def validate_spatial_config(document: Any) -> dict[str, dict[str, tuple[dict, ...]]]:
    """Reject malformed or ambiguous geometry before opening any camera."""
    if (not isinstance(document, dict) or document.keys() != {"schema_version", "cameras"}
            or type(document["schema_version"]) is not int or document["schema_version"] != 1
            or not isinstance(document["cameras"], dict)):
        raise ValueError("Invalid spatial configuration")
    parsed = {}
    for camera_id, values in document["cameras"].items():
        if not isinstance(camera_id, str) or not camera_id.strip() or not isinstance(values, dict):
            raise ValueError("Invalid spatial camera")
        if not values.keys() <= {"inference_rois", "restricted_zones", "tripwires"}:
            raise ValueError("Invalid spatial camera rules")
        result = {}
        for kind in ("inference_rois", "restricted_zones", "tripwires"):
            entries = values.get(kind, [])
            if not isinstance(entries, list):
                raise ValueError("Spatial rules must be lists")
            seen = set()
            converted = []
            for entry in entries:
                common = {"id", "enabled"}
                specific = ({"polygon"} if kind == "inference_rois" else
                            {"polygon", "min_inside_frames", "min_outside_frames"}
                            if kind == "restricted_zones" else
                            {"start", "end", "direction", "stable_frames",
                             "min_displacement", "min_side_distance"})
                name, enabled = _named(entry, common | specific)
                if name in seen:
                    raise ValueError("Spatial IDs must be unique per rule kind")
                seen.add(name)
                rule = {"id": name, "enabled": enabled}
                if kind != "tripwires":
                    rule["polygon"] = _polygon(entry.get("polygon"))
                if kind == "restricted_zones":
                    rule["min_inside_frames"] = _count(entry.get("min_inside_frames"), 2, 2)
                    rule["min_outside_frames"] = _count(entry.get("min_outside_frames"), 2, 1)
                elif kind == "tripwires":
                    rule["start"], rule["end"] = _point(entry.get("start")), _point(entry.get("end"))
                    if rule["start"] == rule["end"]:
                        raise ValueError("Tripwire endpoints must differ")
                    direction = entry.get("direction", "both")
                    if direction not in {"negative_to_positive", "positive_to_negative", "both"}:
                        raise ValueError("Invalid tripwire direction")
                    rule["direction"] = direction
                    rule["stable_frames"] = _count(entry.get("stable_frames"), 3, 3)
                    rule["min_displacement"] = _fraction(entry.get("min_displacement"), 0.03,
                                                          positive=True)
                    rule["min_side_distance"] = _fraction(entry.get("min_side_distance"), 0.005)
                converted.append(rule)
            result[kind] = tuple(converted)
        parsed[camera_id] = result
    return parsed


def load_spatial_config(path: Path) -> "SpatialEngine":
    import yaml

    with Path(path).open(encoding="utf-8") as file:
        return SpatialEngine(yaml.safe_load(file))


@dataclass
class _ZoneTrack:
    inside_count: int = 0
    outside_count: int = 0
    identity_count: int = 0
    kind: str | None = None
    person_id: str | None = None
    last_sample_id: int | None = None
    emitted: bool = False


@dataclass
class _WireTrack:
    frames: int = 0
    last_side: int = 0
    last_side_point: tuple[float, float] | None = None


class SpatialEngine:
    def __init__(self, document: dict):
        self.rules = validate_spatial_config(document)
        self._zones: dict[tuple[str, int, str], _ZoneTrack] = {}
        self._wires: dict[tuple[str, int, str], _WireTrack] = {}

    def reload_rules(self, rules: dict) -> None:
        """Swap in validated rules (e.g. after an in-preview edit) and reset rule state."""
        self.rules = rules
        self._zones.clear()
        self._wires.clear()

    def clear_camera(self, camera_id: str) -> None:
        for states in (self._zones, self._wires):
            for key in tuple(states):
                if key[0] == camera_id:
                    del states[key]

    def _prune(self, camera_id: str, active: set[int]) -> None:
        for states in (self._zones, self._wires):
            for key in tuple(states):
                if key[0] == camera_id and key[1] not in active:
                    del states[key]

    def zone_ids(self, camera_id: str) -> list[str]:
        rules = self.rules.get(camera_id, {})
        return [zone["id"] for zone in rules.get("restricted_zones", ()) if zone["enabled"]]

    def annotate_roi(self, camera_id: str, observations: dict[int, dict],
                     frame_shape: tuple[int, ...]) -> None:
        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            raise ValueError("Spatial rules require a nonempty frame")
        height, width = frame_shape[:2]
        rules = self.rules.get(camera_id, {})
        rois = tuple(rule for rule in rules.get("inference_rois", ()) if rule["enabled"])
        for observation in observations.values():
            bbox = observation.get("bbox")
            if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                    or not all(isinstance(value, (int, float)) and isfinite(value)
                               for value in bbox)
                    or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                observation["in_inference_roi"] = False
                continue
            foot = (((bbox[0] + bbox[2]) / 2) / width, bbox[3] / height)
            observation["in_inference_roi"] = not rois or any(
                _inside(foot, rule["polygon"]) for rule in rois)

    def process(self, camera_id: str, observations: dict[int, dict],
                frame_shape: tuple[int, ...]) -> list[CandidateDraft]:
        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            raise ValueError("Spatial rules require a nonempty frame")
        height, width = frame_shape[:2]
        self._prune(camera_id, set(observations))
        self.annotate_roi(camera_id, observations, frame_shape)
        rules = self.rules.get(camera_id, {"inference_rois": (), "restricted_zones": (),
                                           "tripwires": ()})
        zones = tuple(rule for rule in rules["restricted_zones"] if rule["enabled"])
        wires = tuple(rule for rule in rules["tripwires"] if rule["enabled"])
        candidates = []
        for track_id, observation in observations.items():
            bbox = observation.get("bbox")
            if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                    or not all(isinstance(value, (int, float)) and isfinite(value) for value in bbox)
                    or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                observation["in_inference_roi"] = False
                continue
            foot = (((bbox[0] + bbox[2]) / 2) / width, bbox[3] / height)
            for zone in zones:
                key = (camera_id, track_id, zone["id"])
                memory = self._zones.setdefault(key, _ZoneTrack())
                inside = _inside(foot, zone["polygon"])
                if not inside:
                    memory.outside_count += 1
                    if memory.outside_count >= zone["min_outside_frames"]:
                        memory.inside_count = 0
                        memory.identity_count = 0
                        memory.kind = None
                        memory.person_id = None
                        memory.last_sample_id = None
                        memory.emitted = False
                    continue
                memory.outside_count = 0
                memory.inside_count += 1
                kind = None
                sample_id = observation.get("identity_sample_id")
                person_id = observation.get("person_id")
                if observation.get("identity_evaluated") is True and type(sample_id) is int:
                    if observation.get("identity_status") == "unknown":
                        kind = "unknown_person_in_restricted_zone"
                    elif (observation.get("identity_status") == "registered"
                          and observation.get("zone_access", {}).get(zone["id"]) == "unauthorized"
                          and isinstance(person_id, str)):
                        kind = "unauthorized_zone_entry"
                if kind != memory.kind or (kind and person_id != memory.person_id):
                    memory.identity_count = 0
                    memory.last_sample_id = None
                    memory.emitted = False
                memory.kind = kind
                memory.person_id = person_id if kind else None
                if kind and sample_id != memory.last_sample_id:
                    memory.identity_count += 1
                    memory.last_sample_id = sample_id
                if (kind and not memory.emitted
                        and memory.inside_count >= zone["min_inside_frames"]
                        and (memory.identity_count >= 2
                             or observation.get("identity_confirmed") is True)):
                    memory.emitted = True
                    attributes = {"restricted_zone_id": zone["id"],
                                  "severity": "HIGH", "experimental": True}
                    if kind == "unauthorized_zone_entry":
                        attributes["person_id"] = person_id
                    candidates.append(CandidateDraft(
                        candidate_type=kind, local_track_id=track_id,
                        reason_codes=["stable_restricted_zone_entry", "two_distinct_face_samples",
                                      "experimental_face_threshold"],
                        attributes=attributes,
                    ))
            for wire in wires:
                key = (camera_id, track_id, wire["id"])
                memory = self._wires.setdefault(key, _WireTrack())
                memory.frames += 1
                distance = _cross(wire["start"], wire["end"], foot)
                line_length = hypot(wire["end"][0] - wire["start"][0],
                                    wire["end"][1] - wire["start"][1])
                signed_distance = distance / line_length
                side = (1 if signed_distance > wire["min_side_distance"] else
                        -1 if signed_distance < -wire["min_side_distance"] else 0)
                if side == 0:
                    continue
                previous = memory.last_side
                previous_point = memory.last_side_point
                if (previous and previous != side and previous_point is not None
                        and memory.frames >= wire["stable_frames"]
                        and hypot(foot[0] - previous_point[0], foot[1] - previous_point[1])
                        >= wire["min_displacement"]
                        and _segments_intersect(previous_point, foot, wire["start"], wire["end"])):
                    direction = "negative_to_positive" if previous < side else "positive_to_negative"
                    if wire["direction"] in {direction, "both"}:
                        candidates.append(CandidateDraft(
                            candidate_type="tripwire_crossing", local_track_id=track_id,
                            reason_codes=["stable_directional_tripwire_crossing"],
                            attributes={"tripwire_id": wire["id"], "direction": direction},
                        ))
                memory.last_side, memory.last_side_point = side, foot
        return candidates
