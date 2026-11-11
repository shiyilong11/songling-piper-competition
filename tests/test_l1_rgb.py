"""L1 colour proposals: input contract, hue wrap, masks, ambiguity and clipping."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from piper_sim.l1_rgb import L1RGBDetector, annotate_rgb, load_l1_rgb_config


def fixture():
    image = np.full((240, 360, 3), 130, np.uint8)
    image[40:90, 30:80] = [210, 25, 25]
    image[70:120, 140:190] = [20, 150, 35]
    image[110:160, 250:300] = [220, 190, 15]
    return image


def detect(image, config=None, frame_id="frame0"):
    return L1RGBDetector(config).detect(image, frame_id=frame_id, timestamp_s=1.25)


def test_three_colors_preserve_mask_coordinates_without_mutating_rgb():
    image = fixture()
    original = image.copy()
    result = detect(image)
    assert result.as_dict()["rgb_candidate_counts"] == {"red": 1, "green": 1, "yellow": 1}
    for p, xy in zip(result.proposals, [(30, 40), (140, 70), (250, 110)], strict=True):
        x, y = xy
        expected = np.zeros(image.shape[:2], bool)
        expected[y : y + 50, x : x + 50] = True
        np.testing.assert_array_equal(p.mask, expected)
        assert p.bbox_xywh == (x, y, 50, 50)
        np.testing.assert_allclose(p.centroid_uv, [x + 24.5, y + 24.5])
        assert not p.as_dict()["grasp_ready"] and p.as_dict()["role"] == "unknown"
    preview = annotate_rgb(image, result)
    np.testing.assert_array_equal(image, original)
    assert preview.shape == image.shape
    json.dumps(result.as_dict(), allow_nan=False)


def test_red_hue_wrap_and_multiple_same_color_components():
    hsv = np.zeros((120, 180, 3), np.uint8)
    hsv[20:50, 20:50] = [2, 220, 180]
    hsv[50:90, 110:150] = [178, 220, 180]
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    result = detect(image)
    assert len(result.proposals) == 2
    assert all(p.color == "red" for p in result.proposals)
    assert len({p.instance_id for p in result.proposals}) == 2


@pytest.mark.parametrize("brightness", [0.45, 0.75, 1.0])
def test_simple_global_brightness_variations(brightness):
    image = np.rint(fixture().astype(float) * brightness).astype(np.uint8)
    assert detect(image).as_dict()["rgb_candidate_counts"] == {"red": 1, "green": 1, "yellow": 1}


def test_blue_gray_and_small_noise_are_not_l1_candidates():
    image = np.full((150, 200, 3), 140, np.uint8)
    image[20:80, 30:90] = [20, 30, 220]
    image[100:103, 150:153] = [240, 0, 0]
    assert not detect(image).proposals


def test_clipped_component_is_retained_for_review_not_ready_candidate():
    image = fixture()
    image[180:220, :35] = [220, 20, 20]
    result = detect(image)
    clipped = next(p for p in result.proposals if p.bbox_xywh[0] == 0)
    assert "touches_image_or_roi_boundary" in clipped.rejection_reasons
    assert not clipped.as_dict()["rgb_candidate"]
    assert result.as_dict()["rgb_candidate_counts"]["red"] == 1


def test_target_zone_card_cannot_be_certified_as_a_cube_from_rgb():
    image = np.full((300, 400, 3), 130, np.uint8)
    image[70:220, 100:250] = [220, 190, 15]
    p = detect(image).proposals[0]
    # A square coloured card is intentionally ambiguous until height/size checks.
    assert p.as_dict()["rgb_candidate"]
    assert p.as_dict()["role"] == "unknown"
    assert p.as_dict()["requires_depth_validation"]
    assert not p.as_dict()["grasp_ready"]


def test_roi_exclusions_and_boundary_cuts_are_explicit():
    image = fixture()
    cfg = load_l1_rgb_config()
    cfg["roi"]["include_polygon"] = [[0.15, 0], [1, 0], [1, 1], [0.15, 1]]
    cfg["roi"]["exclude_polygons"] = [[[0.65, 0.4], [0.9, 0.4], [0.9, 0.8], [0.65, 0.8]]]
    result = detect(image, cfg)
    assert {p.color for p in result.proposals} == {"red", "green"}
    red = next(p for p in result.proposals if p.color == "red")
    assert "touches_image_or_roi_boundary" in red.rejection_reasons
    assert not red.mask[:, : round(0.15 * 359)].any()


def test_touching_same_color_is_not_falsely_claimed_as_separate_instances():
    image = np.full((160, 200, 3), 140, np.uint8)
    image[40:80, 40:120] = [200, 20, 20]
    result = detect(image)
    assert len(result.proposals) == 1
    assert "touching" in " ".join(result.proposals[0].as_dict()["warnings"])
    assert not result.as_dict()["block_count_confirmed"]


def test_input_validation_overlapping_ranges_and_frame_local_ids():
    with pytest.raises(ValueError, match="uint8"):
        detect(fixture().astype(float))
    cfg = load_l1_rgb_config()
    cfg["colors"]["green"] = copy.deepcopy(cfg["colors"]["red"])
    with pytest.raises(ValueError, match="overlap"):
        detect(fixture(), cfg)
    before, after = detect(fixture(), frame_id="a"), detect(fixture(), frame_id="b")
    assert not {p.instance_id for p in before.proposals} & {p.instance_id for p in after.proposals}


def test_cli_output_masks_match_json_and_refuses_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    image_path, out = tmp_path / "input.png", tmp_path / "result"
    cv2.imwrite(str(image_path), cv2.cvtColor(fixture(), cv2.COLOR_RGB2BGR))
    cmd = [
        sys.executable,
        str(root / "scripts/detect_l1_rgb.py"),
        "--image",
        str(image_path),
        "--output",
        str(out),
        "--frame-id",
        "test_frame",
        "--timestamp-s",
        "2.5",
    ]
    subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    result = json.loads((out / "detections.json").read_text())
    assert result["timestamp_s"] == 2.5 and not result["grasp_ready"]
    assert result["source"]["mode"] == "saved_image_offline"
    with np.load(out / "masks.npz", allow_pickle=False) as masks:
        for p in result["proposals"]:
            mask = masks[p["mask"]["key"]]
            assert mask.dtype == bool and mask.shape == (240, 360)
            assert int(mask.sum()) == p["area_px"]
    original = (out / "detections.json").read_bytes()
    assert subprocess.run(cmd, cwd=root, capture_output=True, check=False).returncode != 0
    assert (out / "detections.json").read_bytes() == original
