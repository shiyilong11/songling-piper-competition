"""Dynamic blocks and configurable zones; never attach an object to the robot."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

from piper_sim.calibration import ROOT


def load_task(path=None):
    return yaml.safe_load(Path(path or ROOT / "config/traditional_task.yaml").read_text())


def configure_manipulator(root, tcp_offset_m, arm_kp, arm_kv, gripper_kp, gripper_kv):
    """Shared contact/servo setup for the existing demo and multi-object environment."""
    # Servo initial values, not experimentally identified motor parameters.
    for body in root.iter("body"):
        if body.get("name") in [f"link{i}" for i in range(1, 7)] + ["gripper_base"]:
            body.set("gravcomp", "1")
        if body.get("name") in ["link7", "link8"]:
            # Keep CAD appearance; use a rectangular inner pad instead of a convex hull
            # of the entire concave finger. No invisible object attachment or grasp weld.
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            ET.SubElement(
                body,
                "geom",
                name=body.get("name") + "_pad",
                type="box",
                pos="0 -.027 .003",
                size=".022 .023 .003",
                rgba=".15 .15 .15 1",
                friction="1 .01 .001",
                condim="4",
                mass="0",
                solref=".005 1",
            )
    for a in root.find("actuator"):
        arm = int(a.get("joint").replace("joint", "")) <= 6
        a.set("kp", str(arm_kp if arm else gripper_kp))
        a.set("kv", str(arm_kv if arm else gripper_kv))
    root.find(".//site[@name='tcp']").set("pos", f"0 0 {tcp_offset_m}")


def build_task_scene(cfg, mode="l1", seed=0, output=None):
    rng = np.random.default_rng(seed)
    root = ET.parse(ROOT / "assets/piper/scene.xml").getroot()
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("target_") or geom.get("name") == "demo_table":
            world.remove(geom)
    scene = cfg["scene"]
    z = scene["table_top_m"]
    ET.SubElement(
        world,
        "geom",
        name="task_table",
        type="box",
        pos=f".39 -.06 {z / 2}",
        size=f".31 .36 {z / 2}",
        rgba=".58 .6 .62 1",
        friction="1 .01 .001",
    )
    configure_manipulator(
        root,
        cfg["controller"]["grasp_tcp_offset_m"],
        scene["arm_servo_kp"],
        scene["arm_servo_kv"],
        scene["gripper_servo_kp"],
        scene["gripper_servo_kv"],
    )
    zones = ["l1"] if mode == "l1" else ["red", "green", "blue"]
    for name in zones:
        zone = scene["bins"][name]
        xy = zone["center_xy"]
        half = zone["half_size_xy"]
        ET.SubElement(
            world,
            "geom",
            name=f"zone_{name}",
            type="box",
            pos=f"{xy[0]} {xy[1]} {z + 0.0005}",
            size=f"{half[0]} {half[1]} .0005",
            rgba="0.8 .7 .2 .4",
            contype="0",
            conaffinity="0",
            group="2",
        )
    count = 3 if mode == "l1" else 6
    objects = []
    for i in range(count):
        color = ["red", "green", "blue"][i % 3]
        xy = np.array(scene["pick_positions_xy"][i]) + rng.uniform(
            -scene["position_jitter_m"], scene["position_jitter_m"], 2
        )
        yaw = np.deg2rad(rng.uniform(-scene["yaw_jitter_deg"], scene["yaw_jitter_deg"]))
        name = f"cube_{i}"
        half = scene["cube_size_m"] / 2
        b = ET.SubElement(
            world,
            "body",
            name=name,
            pos=f"{xy[0]} {xy[1]} {z + half + 0.001}",
            quat=f"{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}",
        )
        ET.SubElement(b, "freejoint", name=name + "_free")
        ET.SubElement(
            b,
            "geom",
            name=name + "_geom",
            type="box",
            size=f"{half} {half} {half}",
            mass=str(scene["cube_mass_kg"]),
            rgba=" ".join(map(str, scene["colors"][color])),
            friction=" ".join(map(str, scene["friction"])),
            condim="4",
            solref=".005 1",
        )
        objects.append({"name": name, "color": color, "zone": "l1" if mode == "l1" else color})
    output = Path(output or ROOT / "artifacts/traditional" / f"{mode}_seed{seed}.xml")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output)
    mujoco.MjModel.from_xml_path(str(output))
    return output, objects
