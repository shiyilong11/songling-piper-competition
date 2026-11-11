from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from piper_sim.l3_competition import (
    CARDS,
    ExplorationController,
    SurveyView,
    compile_round,
    parse_round,
    preserve_square_evidence,
)


def observed(key, label, x, y):
    return {
        "instance_id": key,
        "identity_status": "estimated",
        "evidence_frame_ids": ["frame_1"],
        "label_hint": label,
        "last_observed_center_base_m": [x, y, 0.05],
    }


def region(color, y):
    transform = np.eye(4)
    transform[:3, 3] = [0.55, y, 0.0]
    return {
        "region_id": f"{color}_region",
        "evidence_frame_ids": ["frame_2"],
        "T_base_region": transform.tolist(),
    }


@pytest.mark.parametrize("numbers", [(5,), (5, 2), (5, 2, 4)])
def test_one_to_three_distinct_published_instructions_are_accepted(numbers):
    assert parse_round("；".join(CARDS[number][0] + "。" for number in numbers)) == numbers


def test_duplicate_or_more_than_three_instructions_are_rejected():
    with pytest.raises(ValueError, match="1 至 3 条"):
        parse_round("\n".join([CARDS[1][0]] * 3))
    with pytest.raises(ValueError, match="1 至 3 条"):
        parse_round("\n".join(CARDS[number][0] for number in (1, 2, 3, 4)))


def test_relative_targets_and_destination_follow_judge_view_in_base_frame():
    snapshot = {
        "objects": [
            observed("a", "persimmon", 0.15, 0.18),
            observed("b", "persimmon", 0.35, -0.18),
            observed("c", "potato", 0.18, 0.11),
            observed("d", "potato", 0.32, -0.11),
        ],
        "regions": [region("red", 0.28), region("green", -0.30), region("yellow", 0.0)],
    }
    plans = compile_round((2, 3), snapshot, [0, 1])
    assert [card["actions"][0]["target_instance_id"] for card in plans] == ["a", "c"]
    assert plans[1]["actions"][0]["destination_region_id"] == "green_region"
    mirrored = compile_round((2, 3), snapshot, [0, -1])
    assert [card["actions"][0]["target_instance_id"] for card in mirrored] == ["b", "d"]
    assert mirrored[1]["actions"][0]["destination_region_id"] == "red_region"


def test_compound_instruction_with_missing_target_stays_unplanned():
    snapshot = {
        "objects": [observed("a", "cube_red", 0.2, 0.1)],
        "regions": [region("red", 0.28), region("yellow", 0.0)],
    }
    simple, compound = compile_round((1, 5), snapshot, [0, 1])
    assert simple["status"] == "planned"
    assert compound["required_actions"] == 2
    assert compound["actions"] == []
    assert compound["verified_actions"] == 0
    assert compound["status"] == "needs_observation"


def test_later_partial_crop_cannot_replace_complete_square_face():
    target = SimpleNamespace(center=np.array([0.16, -0.08, 0.025]))
    cube = observed("green", "cucumber", 0.16, -0.08)
    snapshot = {"objects": [cube]}
    views = {
        "right": SurveyView(np.zeros(6), {"green": ("cube_green", target)}),
        "center": SurveyView(np.zeros(6), {"green": ("cucumber", target)}),
    }
    preserve_square_evidence(snapshot, views)
    assert cube["label_hint"] == "cube_green"
    assert cube["evidence_frame_ids"] == ["right"]


def test_release_verifier_switches_to_another_observed_zone_pose():
    controller = object.__new__(ExplorationController)
    first = np.zeros(6)
    second = np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    controller.zone_views = {"yellow": first}
    controller.zone_observation_poses = {"yellow": [second, first]}
    controller.events = []
    controller.verify_release_from_view = Mock(side_effect=[ValueError("clipped"), "verified"])
    action = {"destination_region_id": "yellow_region", "category": "cube_yellow"}
    assert controller.verify_release(action, None, None) == "verified"
    assert controller.verify_release_from_view.call_args.args[-1] is second
    assert controller.events[0]["reason"] == "clipped"
