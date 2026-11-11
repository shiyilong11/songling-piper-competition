from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from piper_sim.competition_segmentation import (
    CompetitionSegmenter,
    geometry_targets,
    observed_color,
    observed_fruit_category,
    public_geometry_prompts,
    target_from_mask,
)
from piper_sim.observation import Observation


def observation():
    # Downward calibrated camera, 0.5m above a level table. Visible 50mm object.
    rgb = np.full((100, 100, 3), 140, np.uint8)
    depth = np.full((100, 100), 0.5, np.float32)
    rgb[35:65, 35:65] = [200, 10, 10]
    depth[35:65, 35:65] = 0.45
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[:3, 3] = [0.35, 0, 0.56]
    return Observation(
        1.0,
        np.zeros(6),
        np.zeros(6),
        0.07,
        T,
        rgb,
        depth,
        {"fx": 300.0, "fy": 300.0, "cx": 50.0, "cy": 50.0},
    )


def test_prompts_and_metric_geometry_use_only_depth_pixels():
    obs = observation()
    prompts = public_geometry_prompts(obs, np.eye(4), 0.06, {})
    assert len(prompts) == 1
    assert 35 <= prompts[0][0][0] < 65
    mask = np.zeros((100, 100), bool)
    mask[35:65, 35:65] = True
    target = target_from_mask(obs, np.eye(4), 0.06, mask)
    assert target.height == pytest.approx(0.05, abs=1e-5)
    assert observed_color(obs.rgb, target.mask) == "red"


def test_background_mask_and_depth_holes_are_not_valid_objects():
    obs = observation()
    with pytest.raises(ValueError, match="supporting surface"):
        target_from_mask(obs, np.eye(4), 0.06, np.ones((100, 100), bool))
    mask = np.zeros((100, 100), bool)
    mask[35:65, 35:65] = True
    obs.depth_m[35:65, 35:65] = np.nan
    with pytest.raises(ValueError, match="Insufficient registered depth"):
        target_from_mask(obs, np.eye(4), 0.06, mask)


def test_depth_discontinuity_separates_touching_projections_for_verification():
    obs = observation()
    obs.depth_m[35:65, 35:50] = 0.43
    obs.depth_m[35:65, 50:65] = 0.46
    assert len(public_geometry_prompts(obs, np.eye(4), 0.06, {})) == 1
    prompts = public_geometry_prompts(obs, np.eye(4), 0.06, {}, split_depth=True)
    assert len(prompts) == 2
    targets = geometry_targets(obs, np.eye(4), 0.06, split_depth=True)
    assert len(targets) == 2
    assert targets[0].center[0] < 0.35 < targets[1].center[0]


def test_boundary_colours_remain_explicit_and_unknown_is_not_assigned():
    hsv = np.zeros((10, 10, 3), np.uint8)
    hsv[:] = [34, 220, 180]
    hsv[5:] = [36, 220, 180]
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    assert observed_color(rgb, np.ones((10, 10), bool)) == "yellow_green"
    assert observed_color(np.full_like(rgb, 130), np.ones((10, 10), bool)) == "unknown"


@pytest.mark.parametrize(
    "body,green_rows,expected",
    [
        ((22, 162, 186), 0, "potato"),
        ((8, 226, 208), 6, "persimmon"),
        ((26, 195, 243), 1, "bell_pepper"),
    ],
)
def test_compact_fruits_use_skin_and_calyx_evidence(body, green_rows, expected):
    obs = observation()
    hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
    hsv[35:65, 35:65] = body
    hsv[35 : 35 + green_rows, 35:65] = [60, 160, 100]
    obs.rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    mask = np.zeros((100, 100), bool)
    mask[35:65, 35:65] = True
    target = target_from_mask(obs, np.eye(4), 0.06, mask)
    assert observed_fruit_category(obs.rgb, target) == expected


def test_no_automatic_oracle_model_fallback():
    with pytest.raises(ValueError):
        CompetitionSegmenter("oracle")


@pytest.mark.parametrize("label", ["potato", "other", "unknown"])
def test_qwen_categories_are_bound_to_mask_crops(tmp_path, monkeypatch, label):
    obs = observation()
    mask = np.zeros(obs.depth_m.shape, bool)
    mask[35:65, 35:65] = True
    segmenter = CompetitionSegmenter("sam2", classify_fruits=True, category_backend="qwen")
    monkeypatch.setattr(segmenter.sam, "segment", lambda *args: (SimpleNamespace(mask=mask), {}))

    def classify(crops, output):
        assert set(crops) == {"1"}
        with Image.open(crops["1"]) as crop:
            assert crop.size == (228, 180)
        return {"categories": {"1": label}}

    monkeypatch.setattr(segmenter.category_client, "predict_l2_categories", classify)
    candidates, report = segmenter.detect(obs, np.eye(4), 0.06, {}, tmp_path)
    assert [item.color for item in candidates] == [label]
    assert candidates[0].metadata["category_candidate_id"] == "1"
    assert report["category_backend"] == "qwen"
