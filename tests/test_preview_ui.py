"""Preview rendering and in-window editing use synthetic frames and a temporary config."""

import cv2
import numpy as np
import yaml

FRAME = np.zeros((1440, 2560, 3), dtype=np.uint8)
RULES = {
    "restricted_zones": [{"id": "store", "enabled": True,
                          "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.9], [0.1, 0.9]]}],
    "inference_rois": [],
    "tripwires": [{"id": "door", "enabled": True, "start": [0.6, 0.2], "end": [0.9, 0.8],
                   "direction": "both"}],
}


def test_render_tile_draws_after_resize_and_keeps_aspect_ratio():
    from edge.preview_ui import render_tile

    state = {7: {"bbox": [600.0, 300.0, 800.0, 1000.0], "score": 0.91, "identity_status": "unknown"}}
    tile, scale = render_tile(FRAME, "camera_01", state, RULES, width=800, fps=9.5, now=0.0)
    assert tile.shape == (450, 800, 3)
    assert scale == 800 / 2560
    assert tile.any()


def test_track_inside_zone_is_marked_as_alert():
    from edge.preview_ui import ALERT, _track_style, render_tile

    color, parts = _track_style({"identity_status": "registered", "person_id": "P1"}, "Personal_Violence")
    assert color == ALERT and "Personal Violence" in parts
    inside = {1: {"bbox": [500.0, 300.0, 700.0, 1000.0], "score": 0.8}}
    tile, _ = render_tile(FRAME, "camera_01", inside, RULES, width=800, now=0.0)
    assert tile.shape[1] == 800


def _editor(tmp_path, saved):
    from edge.preview_ui import SpatialEditor

    path = tmp_path / "spatial.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "cameras": {}}), encoding="utf-8")
    return SpatialEditor(path, on_saved=lambda: saved.append(True)), path


def test_editor_saves_zone_with_unique_name_and_triggers_reload(tmp_path):
    saved = []
    editor, path = _editor(tmp_path, saved)
    for _ in range(2):
        editor.start("zone", "camera_01")
        for point in [(0.1, 0.1), (0.4, 0.1), (0.4, 0.5)]:
            editor.add_point(point)
        assert editor.save()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    zones = document["cameras"]["camera_01"]["restricted_zones"]
    assert [zone["id"] for zone in zones] == ["zone_1", "zone_2"]
    assert zones[0]["polygon"][1] == [0.4, 0.1]
    assert saved == [True, True] and editor.mode is None


def test_tripwire_autosaves_after_two_points_and_undo_cancel_work(tmp_path):
    editor, path = _editor(tmp_path, [])
    editor.start("roi", "camera_01")
    editor.add_point((0.2, 0.2))
    editor.undo()
    assert editor.points == []
    editor.cancel()
    assert editor.mode is None

    editor.start("tripwire", "camera_02")
    editor.add_point((0.1, 0.5))
    editor.add_point((0.9, 0.5))
    wire = yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"]["camera_02"]["tripwires"][0]
    assert wire["id"] == "tripwire_1" and wire["start"] == [0.1, 0.5]


def test_invalid_polygon_is_rejected_without_writing(tmp_path):
    editor, path = _editor(tmp_path, [])
    before = path.read_text(encoding="utf-8")
    editor.start("zone", "camera_01")
    editor.add_point((0.1, 0.1))
    assert not editor.save()
    assert path.read_text(encoding="utf-8") == before
    assert editor.mode == "zone"  # stays in drawing mode so the operator can continue


def test_window_routes_clicks_to_selected_tile_and_toolbar(tmp_path):
    from edge.preview_ui import TOOLBAR_H, PreviewWindow
    from edge.spatial import SpatialEngine

    path = tmp_path / "spatial.yaml"
    document = {"schema_version": 1, "cameras": {"camera_01": {}, "camera_02": {}}}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    engine = SpatialEngine(document)
    window = PreviewWindow(engine, path)
    frames = {"camera_01": FRAME, "camera_02": FRAME}
    canvas = window.compose(frames, {}, now=0.0)
    assert canvas.shape[0] == TOOLBAR_H + 450 and canvas.shape[1] == 800 * 2 + 4

    window.handle_mouse(cv2.EVENT_LBUTTONDOWN, 1000, TOOLBAR_H + 100)
    assert window.selected == "camera_02"
    zone_button = window.buttons[0]
    window.handle_mouse(cv2.EVENT_LBUTTONDOWN, zone_button.rect[0] + 3, 20)
    assert window.editor.mode == "zone" and window.editor.camera_id == "camera_02"

    focused = window.compose(frames, {}, now=0.0)
    assert focused.shape[1] == 1280
    for x, y in [(100, 100), (600, 100), (600, 500)]:
        window.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, TOOLBAR_H + y)
    window.handle_mouse(cv2.EVENT_RBUTTONDOWN, 600, TOOLBAR_H + 500)
    assert engine.rules["camera_02"]["restricted_zones"][0]["id"] == "zone_1"


def test_editing_is_disabled_without_spatial_engine():
    from edge.preview_ui import PreviewWindow

    window = PreviewWindow(None, None)
    window.handle_key(ord("z"))
    assert window.editor.mode is None
    window.handle_key(ord("q"))
    assert window.should_stop
