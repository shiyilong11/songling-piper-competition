from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from piper_sim.calibration import ROOT
from piper_sim.competition_sensor import backproject
from piper_sim.field_calibration import FieldCalibration


def calibration(tmp_path, parent):
    path = tmp_path / "field_calibration.yaml"
    matrix = np.eye(4)
    matrix[0, 3] = 0.2
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "camera": {
                    "width": 20,
                    "height": 16,
                    "optical_frame": "colour_optical",
                    "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 10.0, "cy": 8.0},
                    "depth": {"meters_per_unit": 0.001, "min_valid_m": 0.1, "max_valid_m": 2.0},
                },
                "hand_eye": {"parent_frame": parent, "T_parent_camera_optical": matrix.tolist()},
            }
        )
    )
    return FieldCalibration(path)


def message(pixels, encoding, *, bigendian=False, frame="colour_optical"):
    rows = [r.tobytes() + b"pad!" for r in pixels]
    return SimpleNamespace(
        width=20,
        height=16,
        step=len(rows[0]),
        is_bigendian=bigendian,
        encoding=encoding,
        data=b"".join(rows),
        header=SimpleNamespace(frame_id=frame),
    )


def test_calibration_template_requires_real_values():
    with pytest.raises(ValueError, match="calibrated camera width"):
        FieldCalibration(ROOT / "deploy/field/field_calibration.yaml")


@pytest.mark.parametrize("parent, expected_x", [("link6", 0.5), ("base_link", 0.2)])
def test_field_decodes_padded_bgr_and_bigendian_depth_in_the_calibrated_frame(
    tmp_path, parent, expected_x
):
    c = calibration(tmp_path, parent)
    bgr = np.full((16, 20, 3), [3, 5, 7], np.uint8)
    depth = np.full((16, 20), 500, dtype=">u2")
    depth[0, 0] = 0
    end = np.eye(4)
    end[0, 3] = 0.3
    obs = c.observation(
        message(bgr, "bgr8"),
        message(depth, "16UC1", bigendian=True),
        np.zeros(7),
        np.zeros(6),
        1.0,
        end,
    )
    cloud, valid = backproject(obs, c.controller_config()["hand_eye"]["T_end_camera"])
    assert obs.rgb[8, 10].tolist() == [7, 5, 3]
    assert np.allclose(cloud[8, 10], [expected_x, 0, 0.5])
    assert not valid[0, 0]


def test_unregistered_depth_frame_is_rejected(tmp_path):
    c = calibration(tmp_path, "link6")
    with pytest.raises(ValueError, match="registered depth"):
        c.observation(
            message(np.zeros((16, 20, 3), np.uint8), "rgb8"),
            message(np.zeros((16, 20), np.uint16), "16UC1", frame="depth_optical"),
            np.zeros(7),
            np.zeros(6),
            0,
            np.eye(4),
        )
