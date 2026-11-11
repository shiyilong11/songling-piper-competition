"""CPU-only indexing and causal geometry features for a first BC baseline.

Explicit allowlists exclude simulator state, teacher plans, RGB and semantic labels
from actor tensors. Stage/outcome are used only for offline filtering/diagnostics.
"""

import hashlib
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import yaml

from piper_sim.calibration import ROOT
from piper_sim.grasp_target_memory import GraspTargetMemory

OBS_KEYS = (
    "target_points",
    "target_valid_mask",
    "scene_points",
    "scene_valid_mask",
    "joint_position_rad",
    "joint_velocity_rad_s",
    "gripper_width_m",
    "T_base_end",
    "previous_action",
    "previous_action_valid",
    "decision_time_s",
    "timestamp_s",
    "table_plane_base",
    "table_valid",
    "table_confidence",
    "T_end_goal",
    "goal_half_size_xy_m",
    "goal_valid",
)
TRANSITION_KEYS = (
    "observation_index",
    "next_observation_index",
    "action",
    "issued_target",
    "action_time_s",
    "stage",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bc_config(path=ROOT / "config/learning/bc_geometry.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["schema_version"] != 1 or cfg["history_steps"] < 1:
        raise ValueError("Unsupported BC schema/history")
    fractions = np.asarray(cfg["split_fractions"], float)
    if fractions.shape != (3,) or not (fractions > 0).all() or not np.isclose(fractions.sum(), 1):
        raise ValueError("Expected three positive split fractions summing to one")
    if cfg["action_protocol"] != "normalized_joint_delta_and_gripper_width_7d":
        raise ValueError("Unsupported action protocol")
    GraspTargetMemory(**cfg["memory"])
    return cfg


def build_manifest(root, cfg):
    """One cohort, whole seed groups; no new claims of asset/category OOD."""
    root = Path(root).resolve()
    entries, skipped, seen = [], Counter(), set()
    for line in (root / "episodes.jsonl").read_text().splitlines():
        item = json.loads(line)
        episode_id = item["episode_id"]
        if episode_id in seen or Path(episode_id).name != episode_id:
            raise ValueError("Duplicate or invalid episode ID")
        seen.add(episode_id)
        if not item["success"] or not item["bc_eligible"]:
            skipped["failed_or_basic_screen_rejected"] += 1
            continue
        path = root / "success" / episode_id
        meta = json.loads((path / "metadata.json").read_text())
        report = json.loads((path / "evaluation.json").read_text())
        if not report["success"] or not report["bc_eligible"]:
            raise ValueError(f"Index/result mismatch: {episode_id}")
        if (
            meta["action_protocol"] != cfg["action_protocol"]
            or meta["policy_hz"] != cfg["policy_hz"]
        ):
            raise ValueError(f"Action contract mismatch: {episode_id}")
        if meta["task"] != "multi_object_selected_target_pick_place":
            raise ValueError("Do not mix task cohorts in this first baseline")
        if report.get("grasp_attempts") != 1:
            skipped["multiple_or_unknown_attempts"] += 1
            continue
        # Same realized initial seed remains in one split even if recorded twice.
        group = f"seed:{meta['candidate_seed']}"
        token = hashlib.sha256(f"{cfg['split_seed']}:{group}".encode()).digest()
        value = int.from_bytes(token[:8], "big") / 2**64
        a, b, _ = cfg["split_fractions"]
        split = "train" if value < a else "validation" if value < a + b else "test"
        entries.append(
            {
                "episode_id": episode_id,
                "relative_path": str(path.relative_to(root)),
                "split": split,
                "group": group,
                "category": meta["category"],
                "layout": meta["layout_id"],
                "source_sha256": meta["source_sha256"],
                "metadata_sha256": digest(path / "metadata.json"),
                "evaluation_sha256": digest(path / "evaluation.json"),
                "transitions_sha256": digest(path / "transitions.npz"),
            }
        )
    entries.sort(key=lambda x: x["episode_id"])
    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "config": cfg,
        "source_index_sha256": digest(root / "episodes.jsonl"),
        "episodes": entries,
        "skipped": dict(skipped),
        "split_counts": dict(Counter(e["split"] for e in entries)),
        "semantics": "development split of known primitive assets; NOT unseen-object OOD",
        "mask_quality_reviewed": False,
        "ready_for_training": False,
    }


def iter_observations(path):
    """Load small public arrays only; RGB/depth payloads stay compressed on disk."""
    for chunk in sorted((Path(path) / "observations").glob("chunk_*.npz")):
        with np.load(chunk, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in OBS_KEYS}
        count = len(arrays["timestamp_s"])
        if any(len(array) != count for array in arrays.values()):
            raise ValueError(f"Mismatched observation arrays: {chunk}")
        for i in range(count):
            yield {key: value[i] for key, value in arrays.items()}


def actor_frame(obs, result, previous_command_target, previous_command_valid):
    """Shared offline/online frame builder. Only past issued commands are inputs."""
    plane = np.asarray(obs["table_plane_base"], float)
    pose = obs["T_base_end"]
    plane_end = np.r_[pose[:3, :3].T @ plane[:3], plane[3] + plane[:3] @ pose[:3, 3]]
    actor = {
        "target_points": result["points_end"],
        "target_valid_mask": result["valid_mask"],
        "target_observed": np.bool_(result["visible"]),
        "target_remembered": np.bool_(result["remembered"]),
        "target_age_s": np.float32(result["age_s"]),
        "target_age_known": np.bool_(result["age_known"]),
        "target_usable": np.bool_(result["usable"]),
        "scene_points": obs["scene_points"],
        "scene_valid_mask": obs["scene_valid_mask"],
        "joint_position_rad": obs["joint_position_rad"],
        "joint_velocity_rad_s": obs["joint_velocity_rad_s"],
        "gripper_width_m": obs["gripper_width_m"],
        "previous_action": obs["previous_action"],
        "previous_action_valid": obs["previous_action_valid"],
        "previous_command_target": np.asarray(previous_command_target, np.float32),
        "previous_command_valid": np.bool_(previous_command_valid),
        "table_plane_end": plane_end.astype(np.float32),
        "table_valid": obs["table_valid"],
        "table_confidence": np.float32(obs["table_confidence"]),
        "T_end_goal": obs["T_end_goal"],
        "goal_half_size_xy_m": obs["goal_half_size_xy_m"],
        "goal_valid": obs["goal_valid"],
    }
    if any(not np.isfinite(v).all() for v in actor.values()):
        raise ValueError("Nonfinite public actor features")
    return actor


def iter_bc_samples(path, cfg):
    """Yield history tensors, labels, and separate diagnostics; never read private/.

    Historical points retain their historical end frames, matching an online
    history buffer. The current remembered target is transformed to current end.
    Closing-command transitions clear static memory. This is conservative command
    evidence, not proof of contact. Fresh measured geometry remains usable.
    """
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta["action_protocol"] != cfg["action_protocol"] or meta["policy_hz"] != cfg["policy_hz"]:
        raise ValueError("Episode/control configuration mismatch")
    with np.load(path / "transitions.npz", allow_pickle=False) as archive:
        transitions = {key: archive[key] for key in TRANSITION_KEYS}
    n = len(transitions["action"])
    if any(len(v) != n for v in transitions.values()):
        raise ValueError("Mismatched transition arrays")
    if not np.array_equal(transitions["observation_index"], np.arange(n)) or not np.array_equal(
        transitions["next_observation_index"], np.arange(1, n + 1)
    ):
        raise ValueError("Expected N transitions aligned to N+1 observations")
    if transitions["action"].shape != (n, 7) or transitions["issued_target"].shape != (n, 7):
        raise ValueError("Expected seven-dimensional action and command state")
    if (
        not np.isfinite(transitions["action"]).all()
        or np.any(np.abs(transitions["action"]) > 1.000001)
        or not np.isfinite(transitions["issued_target"]).all()
    ):
        raise ValueError("Invalid action or command state")
    times = transitions["action_time_s"]
    if not np.isfinite(times).all() or (
        n > 1 and not np.allclose(np.diff(times), 1 / cfg["policy_hz"], atol=1e-6, rtol=0)
    ):
        raise ValueError("Action timestamps do not match the configured policy frequency")
    memory = GraspTargetMemory(**cfg["memory"])
    history = deque(maxlen=cfg["history_steps"])
    previous_closing = False
    count = 0
    for i, obs in enumerate(iter_observations(path)):
        count += 1
        if i > n:
            raise ValueError("Too many observation frames")
        if i == n:
            continue  # terminal observation has no action label
        if (
            not np.isfinite(obs["decision_time_s"])
            or abs(float(obs["decision_time_s"]) - transitions["action_time_s"][i]) > 1e-6
        ):
            raise ValueError("Observation/action timestamp mismatch")
        if i > 0 and (
            not obs["previous_action_valid"]
            or not np.allclose(obs["previous_action"], transitions["action"][i - 1], atol=1e-6)
        ):
            raise ValueError("Previous action must be a past, aligned command")
        closing = bool(obs["previous_action_valid"] and obs["previous_action"][6] < -0.5)
        if closing != previous_closing:
            memory.reset()
        previous_closing = closing
        result = memory.update(
            obs["target_points"],
            obs["target_valid_mask"],
            obs["T_base_end"],
            target_id="selected_target",
            timestamp_s=float(obs["timestamp_s"]),
            now_s=float(obs["decision_time_s"]),
            allow_static_memory=not closing,
        )
        result = memory.express(result, obs["T_base_end"])
        # Camera and robot pose are synchronized in this recorded simulation schema.
        # Live observations must supply the actual image-time and control-time poses.
        actor = actor_frame(
            obs,
            result,
            transitions["issued_target"][i - 1] if i else np.zeros(7, np.float32),
            i > 0,
        )
        history.append(actor)
        stage = str(transitions["stage"][i])
        if stage not in cfg["stages"]:
            continue
        padding = cfg["history_steps"] - len(history)
        items = [history[0]] * padding + list(history)
        yield {
            "actor": {k: np.stack([item[k] for item in items]) for k in actor},
            "history_valid_mask": np.array([False] * padding + [True] * len(history)),
            "action": transitions["action"][i].copy(),
            "diagnostics": {
                "observation_index": i,
                "stage": stage,
                "memory_status": result["status"],
                "target_age_s": result["age_s"],
                "approach_without_geometry": stage in {"pregrasp", "descend"}
                and not result["usable"],
                "command_state_known": i > 0,
                "mask_quality_reviewed": False,
            },
        }
    if count != n + 1:
        raise ValueError(f"Expected {n + 1} observations, got {count}")


def audit_episode(path, cfg):
    stages, states, gaps = Counter(), Counter(), Counter()
    shapes, maximum_age = {}, 0.0
    count = 0
    for sample in iter_bc_samples(path, cfg):
        count += 1
        d = sample["diagnostics"]
        stages[d["stage"]] += 1
        states[d["memory_status"]] += 1
        if d["approach_without_geometry"]:
            gaps[d["stage"]] += 1
        if d["memory_status"] == "remembered":
            maximum_age = max(maximum_age, d["target_age_s"])
        shapes = {k: list(v.shape) for k, v in sample["actor"].items()}
    return {
        "episode_id": Path(path).name,
        "samples": count,
        "stages": dict(stages),
        "memory_status": dict(states),
        "approach_without_geometry": dict(gaps),
        "maximum_used_memory_age_s": maximum_age,
        "actor_shapes": shapes,
        "ready_for_training": False,
        "note": "Structural/causal check only; masks and missing-geometry handling need review.",
    }
